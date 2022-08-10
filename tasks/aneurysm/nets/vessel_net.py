from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import OrderedDict
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce
from .swinTFPN import swintfpn
from .resnetFPN import resnetfpn

from .resunet import norm, CB, CBR, BasicBlock, DownSample, DANetHead, DACBlock, SPPBlock 

class SimpleNet(nn.Module):
    def __init__(self, segClasses = 2, k=16):
        
        super(SimpleNet, self).__init__()
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, k, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True))]))
        
        self.layer1 = nn.Sequential(
            CBR(k, k),
            CBR(k, 2*k),
            CBR(2*k, 2*k, dilation=2),
            CBR(2*k, 2*k, dilation=4),
            CBR(2*k, 2*k, dilation=8),
            CBR(2*k, 4*k),
            CBR(4*k, 4*k),
            nn.Conv3d(4*k, segClasses, kernel_size=3, bias=False),
        )
    
    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = F.interpolate(x, x_size[2:], mode='trilinear', align_corners=True)
        return {'y': x}

class DAResNet18(nn.Module):   
    def __init__(self, segClasses = 2, k=16, input_channels=1):
        
        super(DAResNet18, self).__init__()
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(input_channels, k, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True))]
        ))
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock,    k, 2, kernel_size=(3,3,3), stride=1)
        self.layer2 = self._make_layer(BasicBlock,  2*k, 2, kernel_size=(3,3,3), stride=2)
        self.layer3 = self._make_layer(BasicBlock,  4*k, 2, kernel_size=(3,3,3), stride=(2,2,2))
        self.layer4 = self._make_layer(BasicBlock,  8*k, 2, kernel_size=(1,3,3), stride=(1,2,2))
        
        self.class4 = DANetHead(8*k, 8*k)
        
        self.class3 = nn.Sequential(
            #CBR(4*k+8*k, 4*k, (3,3,3))
            BasicBlock(4*k+8*k, 4*k, kernel_size=3)
        )

        self.class2 = nn.Sequential(
            #CBR(2*k+4*k, 2*k, (3,3,3)),
            BasicBlock(2*k+4*k, 2*k, kernel_size=3)
        )
        
        self.class1 = nn.Sequential(
            #CBR(k+2*k, 2*k),
            BasicBlock(k+2*k, 2*k, kernel_size=3),
            nn.Conv3d(2*k, segClasses, kernel_size=3, bias=False),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        
        x_size = x.size()


        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        d4 = self.class4(self.layer4(e3))
        
        d3 = self.class3(torch.cat([e3,
                         F.interpolate(d4, e3.size()[2:], mode='trilinear', align_corners=True)], 1))        
        d2 = self.class2(torch.cat([e2,
                         F.interpolate(d3, e2.size()[2:], mode='trilinear', align_corners=True)], 1))
        d1 = self.class1(torch.cat([e1,
                         F.interpolate(d2, e1.size()[2:], mode='trilinear', align_corners=True)], 1))
        
        x = F.interpolate(d1, x_size[2:], mode='trilinear', align_corners=True)
        
        return {'y': x}

    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = CB(out_channels, 1)

    def forward(self, x):
        x=self.conv(x)
        x=F.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = CBR(out_channels, out_channels//2)
        self.conv2 = CB(out_channels//2, out_channels)
    
    def forward(self, x):
        x = nn.AvgPool3d(x.size()[2:])(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x=F.sigmoid(x)
        return x

class Decoder(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = CBR(in_channels, channels)
        self.conv2 = CBR(channels, out_channels)
        self.spatial_gate = sSE(out_channels)
        # self.channel_gate = cSE(out_channels)

        self.dropout = nn.Dropout3d(0.05, False)

    def forward(self, x, e=None):
        if e is not None:
            x = torch.cat([e,
                   F.interpolate(x, e.size()[2:], mode='trilinear', align_corners=True)], 1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        g1 = self.spatial_gate(x)
        # g2 = self.channel_gate(x)
        
        # x = g1 * x + g2 * x
        x = g1 * x
        return self.dropout(x)
                         
class DASEResNet18(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, input_channels=1, attention=True, heatmap=False):  
        
        super(DASEResNet18, self).__init__()
        self.attention = attention
        self.heatmap = heatmap
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(input_channels, k, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True))]
        ))
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock,    k, 2, kernel_size=(3,3,3), stride=1)
        self.layer2 = self._make_layer(BasicBlock,  2*k, 2, kernel_size=(3,3,3), stride=2)
        self.layer3 = self._make_layer(BasicBlock,  4*k, 2, kernel_size=(3,3,3), stride=(2,2,2))
        self.layer4 = self._make_layer(BasicBlock,  8*k, 2, kernel_size=(3,3,3), stride=(2,2,2))
        
        if attention:
            self.class4 = DANetHead(8*k, 8*k)
        
        self.class3 = Decoder(8*k+4*k, 4*k, k)
        self.class2 = Decoder(2*k+k, 2*k, k)
        self.class1 = Decoder(  k+k, 2*k, k)
        
        self.channel1 = nn.Conv3d(96, 24, kernel_size=1, stride=1, bias=False)
        self.channel2 = nn.Conv3d(192, 48, kernel_size=1, stride=1, bias=False) #48
        self.channel3 = nn.Conv3d(384, 96, kernel_size=1, stride=1, bias=False)
        self.channel4 = nn.Conv3d(384, 192, kernel_size=1, stride=1, bias=False) #192
        
        
        
        self.class0 = nn.Sequential(
            nn.Conv3d(4*k, 2*k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv3d(2*k, segClasses, kernel_size=1, bias=False)
        )

        if self.heatmap:
            self.class_hp = nn.Sequential(
                nn.Conv3d(4*k, 2*k, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ELU(inplace=True),
                nn.Conv3d(2*k, 1, kernel_size=1, bias=False),
                #nn.ReLU(inplace=True),
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _upsample_time(self, x, outsize):
        _,_,t, h, w = x.size()
        # print('time', x.shape)
        x_upsampled = F.interpolate(x, outsize, mode='trilinear')  #(x, [t*2, h, w]

        return x_upsampled
                
    def forward(self, x):

        x_size = x.size()
        
#####################   swin transformer  ######################
      
        swintf = swintfpn()
        swintf.cuda()
        
        x_swin, x_list = swintf(x)
        x_swin1 = x_list[0]
        x_swin2 = x_list[1]  #stage 2
        x_swin3 = x_list[2]  #stage 3  
        x_swin4 = x_swin
                
        x_swin1 = self.channel1(x_swin1)
        x_swin2 = self.channel2(x_swin2)
        x_swin3 = self.channel3(x_swin3)
        x_swin4 = self.channel4(x_swin4)
        
        x_swin1 = self._upsample_time(x_swin1,[128, 128, 128])
        x_swin2 = self._upsample_time(x_swin2, [64, 64, 64]) 
        x_swin3 = self._upsample_time(x_swin3, [32, 32, 32])
        x_swin = self._upsample_time(x_swin4, [16, 16, 16]) 
        
        x_swin1 = 0.3*x_swin1 
        x_swin2 = 0.3*x_swin2
        x_swin3 = 0.3*x_swin3  #0.3
        x_swin = 0.5*x_swin
        


#####################   swin transformer  ######################
        e0 = self.layer0(x)
        e1_0 = self.layer1(e0)
        e1 = e1_0 + x_swin1 
        e2_0 = self.layer2(e1)
        e2 = e2_0 + x_swin2#
        e3 = self.layer3(e2)
        e3_0 = self.layer3(e2)
        e3 = e3_0 + x_swin3 

        if self.attention:
            d4_0 = self.layer4(e3) 
        else:
            d4_0 = self.layer4(e3)

        d4 = d4_0 + x_swin   
        d3 = self.class3(d4, e3)
        d2 = self.class2(d3, e2)        
        d1 = self.class1(d2, e1) #(1,24,128,128,128)

        
        
        feat = torch.cat([
                e0,
                d1,
                F.interpolate(d2, e0.size()[2:], mode='trilinear', align_corners=True),
                F.interpolate(d3, e0.size()[2:], mode='trilinear', align_corners=True)], 1)
        
        seg = self.class0(feat)
        seg = F.interpolate(seg, x_size[2:], mode='trilinear', align_corners=True)
        
        if self.heatmap:
            hp = self.class_hp(feat)
            hp = F.interpolate(hp, x_size[2:], mode='trilinear', align_corners=True)
            return {'y': seg, 'hp': hp}
        
        return {'y': seg}

    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

class DASEResNet18_SDM(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, input_channels=1):
        
        super(DASEResNet18_SDM, self).__init__()
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(input_channels, k, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True))]
        ))
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock,    k, 2, kernel_size=(3,3,3), stride=1)
        self.layer2 = self._make_layer(BasicBlock,  2*k, 2, kernel_size=(3,3,3), stride=2)
        self.layer3 = self._make_layer(BasicBlock,  4*k, 2, kernel_size=(3,3,3), stride=(2,2,2))
        self.layer4 = self._make_layer(BasicBlock,  8*k, 2, kernel_size=(1,3,3), stride=(1,2,2))
        
        self.class4 = DANetHead(8*k, 8*k)
        
        self.class3 = Decoder(8*k+4*k, 4*k, k)
        self.class2 = Decoder(2*k+k, 2*k, k)
        self.class1 = Decoder(  k+k, 2*k, k)
        
        self.class0 = nn.Sequential(
            nn.Conv3d(4*k, 2*k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv3d(2*k, segClasses-1, kernel_size=1, bias=False),
            nn.Tanh()
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        x_size = x.size()
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        d4 = self.class4(self.layer4(e3))
        
        d3 = self.class3(d4, e3)
        d2 = self.class2(d3, e2)
        d1 = self.class1(d2, e1)
        
        x = self.class0(torch.cat([
                e0,
                d1,
                F.interpolate(d2, e0.size()[2:], mode='trilinear', align_corners=True),
                F.interpolate(d3, e0.size()[2:], mode='trilinear', align_corners=True)], 1))
        
        x = F.interpolate(x, x_size[2:], mode='trilinear', align_corners=True)
        
        return {'y': x}

    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

class DASEResNet34(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, input_channels=1, drop_rate=0.0):
        
        super(DASEResNet34, self).__init__()
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(input_channels, k, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True))]
        ))
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock,    k, 3, kernel_size=(3,3,3), stride=1)
        self.layer2 = self._make_layer(BasicBlock,  2*k, 4, kernel_size=(3,3,3), stride=2)
        self.layer3 = self._make_layer(BasicBlock,  4*k, 6, kernel_size=(3,3,3), stride=(2,2,2))
        self.layer4 = self._make_layer(BasicBlock,  8*k, 3, kernel_size=(1,3,3), stride=(1,2,2))
        
        self.class4 = DANetHead(8*k, 8*k)
        
        self.class3 = Decoder(8*k+4*k, 4*k, k)
        self.class2 = Decoder(2*k+k, 2*k, k)
        self.class1 = Decoder(  k+k, 2*k, k)
        
        self.class0 = nn.Sequential(
            nn.Conv3d(4*k, 2*k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv3d(2*k, segClasses, kernel_size=1, bias=False)
        )
        
        self.drop_rate = drop_rate
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        x_size = x.size()
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        d4 = self.class4(self.layer4(e3))
        if self.drop_rate > 0:
            d4 = F.dropout(d4, p=self.drop_rate, training=self.training)
        
        d3 = self.class3(d4, e3)
        d2 = self.class2(d3, e2)
        d1 = self.class1(d2, e1)
        
        x = self.class0(torch.cat([
                e0,
                d1,
                F.interpolate(d2, e0.size()[2:], mode='trilinear', align_corners=True),
                F.interpolate(d3, e0.size()[2:], mode='trilinear', align_corners=True)], 1))
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
            
        x = F.interpolate(x, x_size[2:], mode='trilinear', align_corners=True)
        
        return {'y': x}

    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

class ResNet_UNet(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, input_channels=1, attention=True, pool=True):
        
        super(ResNet_UNet, self).__init__()
        self.pool = pool
        self.attention = attention
        self.inplanes = k
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(input_channels, k, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True))]
        ))        
        self.layer1 = self._make_layer(BasicBlock,     k,  3, kernel_size=(3,3,3), stride=1)
        
        if pool:
            self.layer2 = nn.Sequential(
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                self._make_layer(BasicBlock,  2*k, 4, kernel_size=(3,3,3), stride=1)
            )
            self.layer3 = nn.Sequential(
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                self._make_layer(BasicBlock,  4*k, 4, kernel_size=(3,3,3), stride=1)
            )
        else:
            self.layer2 = self._make_layer(BasicBlock,  2*k, 4, kernel_size=(3,3,3), stride=2)
            self.layer3 = self._make_layer(BasicBlock,  4*k, 6, kernel_size=(3,3,3), stride=2)
        
        if attention:
            self.layer4 = nn.Sequential(
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0,1,1)),
                DANetHead(4*k, 4*k)
            )
        
        if attention:
            self.class2 = nn.Sequential(
                BasicBlock(4*k + 4*k+2*k, 4*k, kernel_size=3)
            )
        else:
            self.class2 = nn.Sequential(
                BasicBlock(4*k + 2*k, 4*k, kernel_size=3)
            )

        self.class1 = nn.Sequential(
            BasicBlock(2*k+4*k, 2*k, kernel_size=3),
            nn.Conv3d(2*k, segClasses, kernel_size=3, bias=False)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
   
    def forward(self, x):
        
        x_size = x.size()
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        
        if self.attention:
            e4 = F.interpolate(self.layer4(e3), e3.size()[2:], mode='trilinear', align_corners=True)
            d2 = self.class2(torch.cat([
                F.interpolate(e4, e2.size()[2:], mode='trilinear', align_corners=True),
                F.interpolate(e3, e2.size()[2:], mode='trilinear', align_corners=True),
                e2], 1))
        else:
            d2 = self.class2(torch.cat([
                F.interpolate(e3, e2.size()[2:], mode='trilinear', align_corners=True),
                e2], 1))
            
        d1 = self.class1(torch.cat([
            F.interpolate(d2, e1.size()[2:], mode='trilinear', align_corners=True),
            e1, e0], 1))
        
        x = F.interpolate(d1, x_size[2:], mode='trilinear', align_corners=True)
        
        return {'y': x}
    
    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []                             
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

class Attention_block(nn.Module):
    """Attetion block
    """
    
    def __init__(self, F_g, F_l, F_v):
        super(Attention_block, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_v, kernel_size=1, stride=1, padding=0, bias=False),
            norm(F_v)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_v, kernel_size=1, stride=1, padding=0, bias=False),
            norm(F_v)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_v, 1, kernel_size=1, stride=1, padding=0, bias=False),
            norm(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class DAMultiHeadResNet(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, input_channels=1, drop_rate=0.0):
        super(DAMultiHeadResNet, self).__init__()
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(input_channels, k, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(k, 2*k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', norm(2*k)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv3d(2*k, 2*k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn3', norm(2*k)),
            ('relu3', nn.ReLU(inplace=True))]
        ))
        self.inplanes = 2*k
        self.layer1 = self._make_layer(BasicBlock,  2*k, 3, kernel_size=(3,3,3), stride=2)
        self.layer2 = self._make_layer(BasicBlock,  4*k, 4, kernel_size=(3,3,3), stride=2)
        self.layer3 = self._make_layer(BasicBlock,  8*k, 6, kernel_size=(1,3,3), stride=(1,2,2))
        self.layer4 = self._make_layer(BasicBlock, 16*k, 3, kernel_size=(1,3,3), stride=(1,2,2))
        
        self.da_block = DANetHead(16*k, 16*k)
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2)), 
            nn.Conv3d(16*k, 8*k, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            norm(8*k),
            nn.ReLU(inplace=True)
        )
        self.att3 = Attention_block(8*k, 8*k, 8*k)
        self.class3 = nn.Sequential(
            CBR(8*k+8*k, 8*k, (1,3,3)),
            CBR(8*k, 8*k, (1,3,3))
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2)), 
            nn.Conv3d(8*k, 4*k, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            norm(4*k),
            nn.ReLU(inplace=True)
        )
        self.att2 =  Attention_block(4*k, 4*k, 4*k)
        self.class2 = nn.Sequential(
            CBR(4*k+4*k, 4*k, (3,3,3)),
            CBR(4*k, 4*k, (3,3,3))
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2,2)), 
            nn.Conv3d(4*k, 2*k, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False),
            norm(2*k),
            nn.ReLU(inplace=True)
        )
        self.att1 =  Attention_block(2*k, 2*k, 2*k)
        
        self.class1 = nn.Sequential(
            CBR(2*k+2*k, 2*k, (3,3,3)),
            CBR(2*k, 2*k, (3,3,3))
        )
        
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2,2)), 
            nn.Conv3d(2*k, 2*k, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False),
            norm(2*k),
            nn.ReLU(inplace=True)
        )
        self.att0 =  Attention_block(2*k, 2*k, 2*k)        
        self.class0 = nn.Sequential(
            CBR(2*k+2*k, 2*k, (3,3,3)),
            CBR(2*k, 2*k, (3,3,3))
        )
        
        self.seg =   nn.Sequential(
                            nn.Conv3d(2*k, segClasses-1, kernel_size=1, stride=1, padding=0),
                            nn.Sigmoid())
        self.sdm = nn.Sequential(
                            nn.Conv3d(2*k, segClasses-1, kernel_size=1, stride=1, padding=0),
                            nn.Tanh())
    
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    
    def forward(self, x):
        
        x_size = x.size()
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.da_block(self.layer4(e3))
        
        d3 = self.up3(e4)
        d3 = self.class3(torch.cat((self.att3(d3, e3), d3), dim=1))
        
        d2 = self.up2(d3)
        d2 = self.class2(torch.cat((self.att2(d2, e2), d2), dim=1))
        
        d1 = self.up1(d2)
        d1 = self.class1(torch.cat((self.att1(d1, e1), d1), dim=1))
        
        d0 = self.up0(d1)
        d0 = self.class0(torch.cat((self.att0(d0, e0), d0), dim=1))
        
        seg = self.seg(d0)
        sdm = self.sdm(d0)
        
        seg = F.interpolate(seg, x_size[2:], mode='trilinear', align_corners=True)
        sdm = F.interpolate(sdm, x_size[2:], mode='trilinear', align_corners=True)
        
        return {'y': seg.squeeze(1), 'sdm': sdm.squeeze(1)}
        
    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)
