
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce

from .oc_block import BaseOC_Module
from .pyramid_oc_block import Pyramid_OC_Module
from .attention import PAM_Module, CAM_Module
from utils.config import cfg

def norm(planes, mode='bnfix', groups=12):
    if mode == 'bn':
        return nn.BatchNorm3d(planes, momentum=0.95, eps=1e-03)
    if mode == 'bnfix': # bn static
        return nn.BatchNorm3d(planes, momentum=0.95, eps=1e-03, track_running_stats=False)
    elif mode == 'gn':
        return nn.GroupNorm(groups, planes)
    elif mode == 'in':
        return nn.InstanceNorm3d(planes)#, affine=True, track_running_stats=True)
    elif mode == 'lrn':
        return nn.LocalResponseNorm(groups)
    else:
        return nn.Sequential()

class CBR(nn.Module):
    
    def __init__(self, nIn, nOut, kSize=(3,3,3), stride=1, dilation=1):
        super(CBR, self).__init__()
        
        if not isinstance(kSize, tuple):
            kSize = (kSize, kSize, kSize)
        
        padding = (int((kSize[0]-1)/2) * dilation, int((kSize[1]-1)/2) * dilation, int((kSize[2]-1)/2) * dilation)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, 
                              bias=False, dilation=dilation)
        self.bn = norm(nOut)
        self.act = nn.ReLU(True)
        
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        
        return output

class CB(nn.Module):
    
    def __init__(self, nIn, nOut, kSize=(3,3,3), stride=1, dilation=1):
        super(CB, self).__init__()
        
        if not isinstance(kSize, tuple):
            kSize = (kSize, kSize, kSize)
            
        padding = (int((kSize[0]-1)/2) * dilation, int((kSize[1]-1)/2) * dilation, int((kSize[2]-1)/2) * dilation)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, 
                              bias=False, dilation=dilation)
        if nOut == 1:
            self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)
        else:
            self.bn = norm(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        
        return output

class C(nn.Module):
    def __init__(self, nIn, nOut, kSize=(3,3,3), stride=1, dilation=1):
        super(C, self).__init__()
        padding = (int((kSize[0]-1)/2) * dilation, int((kSize[1]-1)/2) * dilation, int((kSize[2]-1)/2) * dilation)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=False, dilation=dilation)
    def forward(self, input):
        return self.conv(input)
        
class BR(nn.Module):
    def __init__(self, nIn):
        super(BR, self).__init__()

        self.bn = norm(nIn)
        self.act = nn.ReLU(True)
    def forward(self, input):
        return self.act(self.bn(input))

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, nIn, nOut, kernel_size=(3,3,3), prob=0.03, stride=1, dilation=1):
        
        super(BasicBlock, self).__init__()
        
        self.c1 = CBR(nIn, nOut, kernel_size, stride, dilation)
        self.c2 = CB(nOut, nOut, kernel_size, 1, dilation)
        self.act = nn.ReLU(True)
        
        self.downsample=None
        if nIn != nOut or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(nIn, nOut, kernel_size=1, stride=stride, bias=False),
                norm(nOut)
            )
            
    def forward(self, input):
        output = self.c1(input)
        output = self.c2(output)
        if self.downsample is not None:
            input = self.downsample(input)

        output = output + input
        output = self.act(output)

        return output
    
class PSPModule(nn.Module):
    
    def __init__(self, nIn, nOut, sizes=(1,2,3,6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(nIn, size) for size in sizes])
        self.bottleneck = nn.Conv3d(nIn * (len(sizes) + 1), nOut, kernel_size=1, bias=False)
        self.relu = nn.ReLU(True)
    
    def _make_stage(self, nIn, size):
        prior = nn.AdaptiveAvgPool3d(output_size=size)
        conv = nn.Conv3d(nIn, nIn, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)
    
    def forward(self, x):
        h, w, d = x.size(2), x.size(3), x.size(4)
        priors = [F.upsample(input=stage(x), size=(h,w,d), mode='trilinear') for stage in self.stages] + [x]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class DownSample(nn.Module):
    def __init__(self, nIn, nOut, pool='max'):
        super(DownSample, self).__init__()
        
        if pool == 'conv':
            self.pool = CBR(nIn, nOut, 3, 2)
        elif pool == 'max':
            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.pool = pool
            if nIn != nOut:
                self.pool = nn.Sequential(pool, CBR(nIn, nOut, 1, 1))
    
    def forward(self, input):
        output = self.pool(input)
        return output
    
class Upsample(nn.Module):
    
    def __init__(self, nIn, nOut):
        super(Upsample, self).__init__()
        self.conv = CBR(nIn, nOut)
        
    def forward(self, x):
        p = F.upsample(x, scale_factor=2, mode='trilinear')
        return self.conv(p)

class ResUNet(nn.Module):
    
    def __init__(self, num_classes = 2, k=16, psp=True):
        
        super(ResUNet, self).__init__()
        
        self.layer0 = CBR(1, k, 7, 1)
        self.class0 = nn.Sequential(
            BasicBlock(k+2*k, 2*k),
            nn.Conv3d(2*k, num_classes, kernel_size=1, bias=False)
        )
        
        self.pool1 = DownSample(k, k, 'max')
        self.layer1 = nn.Sequential(
            BasicBlock(k, 2*k),
            BasicBlock(2*k, 2*k)
        )
        #self.br_1 = BR(k+2*k)
        self.class1 = nn.Sequential(
            BasicBlock(2*k+4*k, 4*k),
            CBR(4*k, 2*k, 1)
        )
        
        self.pool2 = DownSample(2*k, 2*k, 'max')
        self.layer2 = nn.Sequential(
            BasicBlock(2*k, 4*k),
            BasicBlock(4*k, 4*k)
        )
        
        self.class2 = nn.Sequential(
            BasicBlock(4*k+8*k, 8*k),
            CBR(8*k, 4*k, 1)
        )
        
        self.pool3 = DownSample(4*k, 4*k, 'max')
        self.layer3 = nn.Sequential(
            BasicBlock(4*k, 8*k, dilation=1),
            BasicBlock(8*k, 8*k, dilation=2),
            BasicBlock(8*k, 8*k, dilation=4)
        )
        #self.br_3 = BR(7*k+8*k)
        sizes=((1,1,1), (2,2,2), (3, 3, 3), (6, 6, 6))
        self.class3 = PSPModule(8*k, 8*k, sizes) if psp else CBR(8*k, 8*k, 1)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear')
        
    def forward(self, x):
        
        output0 = self.layer0(x)
        output1_0 = self.pool1(output0)
        output1 = self.layer1(output1_0)
        
        output2_0 = self.pool2(output1)
        output2 = self.layer2(output2_0)
        
        output3_0 = self.pool3(output2)
        output3 = self.layer3(output3_0)
        
        output = self.class3(output3)
        output = self.up3(output)
        output = self.class2(torch.cat([output2, output], 1))
        output = self.up2(output)
        output = self.class1(torch.cat([output1, output], 1))
        output = self.up1(output)
        output = self.class0(torch.cat([output0, output], 1))
        
        return {'y': output }

class OCResUNet(nn.Module):
    
    def __init__(self, num_classes = 2, k=16, psp=True):
        
        super(OCResUNet, self).__init__()
        
        self.layer0 = CBR(1, k, 7, 1)
        self.class0 = nn.Sequential(
            BasicBlock(k+2*k, 2*k),
            nn.Conv3d(2*k, num_classes, kernel_size=1, bias=False)
        )
        
        self.pool1 = DownSample(k, k, 'max')
        self.layer1 = nn.Sequential(
            BasicBlock(k, 2*k),
            BasicBlock(2*k, 2*k)
        )
        
        self.class1 = nn.Sequential(
            BasicBlock(2*k+4*k, 4*k),
            CBR(4*k, 2*k, 1)
        )
        
        self.pool2 = DownSample(2*k, 2*k, 'max')
        self.layer2 = nn.Sequential(
            BasicBlock(2*k, 4*k),
            BasicBlock(4*k, 4*k)
        )
        
        self.class2 = nn.Sequential(
            BasicBlock(4*k+8*k, 8*k),
            CBR(8*k, 4*k, 1)
        )
        
        self.pool3 = DownSample(4*k, 4*k, 'max')
        self.layer3 = nn.Sequential(
            BasicBlock(4*k, 8*k, dilation=1),
            BasicBlock(8*k, 8*k, dilation=2),
            BasicBlock(8*k, 8*k, dilation=4)
        )
        sizes=((1,1,1), (2,2,2), (3, 3, 3), (6, 6, 6))
        if psp:
            self.class3 = Pyramid_OC_Module(in_channels=8*k, out_channels=8*k, dropout=0.05, sizes=([1, 2, 3, 6]))
        else:
            self.class3 = BaseOC_Module(in_channels=8*k, out_channels=8*k, key_channels=4*k, value_channels=4*k,
                                         dropout=0.05, sizes=([1]))

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear')
        
    def forward(self, x):
        
        output0 = self.layer0(x)
        output1_0 = self.pool1(output0)
        output1 = self.layer1(output1_0)
        
        output2_0 = self.pool2(output1)
        output2 = self.layer2(output2_0)
        
        output3_0 = self.pool3(output2)
        output3 = self.layer3(output3_0)
        
        output = self.class3(output3)
        output = self.up3(output)
        output = self.class2(torch.cat([output2, output], 1))
        output = self.up2(output)
        output = self.class1(torch.cat([output1, output], 1))
        output = self.up1(output)
        output = self.class0(torch.cat([output0, output], 1))
        
        return {'y': output}

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1), 
                                  nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                  nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout3d(0.05, False), nn.Conv3d(inter_channels, out_channels, 1),
                                  nn.ReLU())

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output

class DAResUNet(nn.Module):
    
    def __init__(self, num_classes = 2, k=16, psp=True):
        
        super(DAResUNet, self).__init__()
        
        self.layer0 = CBR(1, k, 7, 1)
        self.class0 = nn.Sequential(
            BasicBlock(k+2*k, 2*k),
            nn.Conv3d(2*k, num_classes, kernel_size=1, bias=False)
        )
        
        self.pool1 = DownSample(k, k, 'max')
        self.layer1 = nn.Sequential(
            BasicBlock(k, 2*k),
            BasicBlock(2*k, 2*k)
        )
        
        self.class1 = nn.Sequential(
            BasicBlock(2*k+4*k, 4*k),
            CBR(4*k, 2*k, 1)
        )
        
        self.pool2 = DownSample(2*k, 2*k, 'max')
        self.layer2 = nn.Sequential(
            BasicBlock(2*k, 4*k),
            BasicBlock(4*k, 4*k)
        )
        
        self.class2 = nn.Sequential(
            BasicBlock(4*k+8*k, 8*k),
            CBR(8*k, 4*k, 1)
        )
        
        self.pool3 = DownSample(4*k, 4*k, 'max')
        self.layer3 = nn.Sequential(
            BasicBlock(4*k, 8*k, dilation=1),
            BasicBlock(8*k, 8*k, dilation=2),
            BasicBlock(8*k, 8*k, dilation=4)
        )
        
        sizes=((1,1,1), (2,2,2), (3, 3, 3), (6, 6, 6))
        self.class3 = DANetHead(8*k, 8*k)
        

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear')

        self._init_weight()
        
    def  _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        
        output0 = self.layer0(x)
        output1_0 = self.pool1(output0)
        output1 = self.layer1(output1_0)
        
        output2_0 = self.pool2(output1)
        output2 = self.layer2(output2_0)
        
        output3_0 = self.pool3(output2)
        output3 = self.layer3(output3_0)
        
        output = self.class3(output3)
        output = self.up3(output)
        output = self.class2(torch.cat([output2, output], 1))
        output = self.up2(output)
        output = self.class1(torch.cat([output1, output], 1))
        output = self.up1(output)
        output = self.class0(torch.cat([output0, output], 1))
        
        return {'y': output}
    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 12, 24, 36]):
        super(ASPP, self).__init__()
        
        k = in_channels // 4
        self.aspp1 = CBR(in_channels, k, 1, 1, dilations[0])
        self.aspp2 = CBR(in_channels, k, 3, 1, dilations[1])
        self.aspp3 = CBR(in_channels, k, 3, 1, dilations[2])
        self.aspp4 = CBR(in_channels, k, 3, 1, dilations[3])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                             nn.Conv3d(in_channels, k, 1, stride=1, bias=False),
                                             norm(k),
                                             nn.ReLU())
        self.conv = CBR(5*k, out_channels, 1, 1)
        self.dropout = nn.Dropout(0.5)
        
        self._init_weight()
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv(x)
        
        return self.dropout(x)
    
    def  _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
class FCN(nn.Module):
    
    def __init__(self, num_classes = 2, k=8, psp=True):
        super(FCN, self).__init__()
        
        self.layer0 = nn.Sequential(
            BasicBlock(1, k),
            BasicBlock(k, k)
        )
        
        self.layer1 = nn.Sequential(
            BasicBlock(k, 2*k),
            BasicBlock(2*k, 2*k)
        )
        
        self.layer2 = nn.Sequential(
            BasicBlock(2*k, 4*k),
            BasicBlock(4*k, 4*k)
        )
        
        self.layer3 = nn.Sequential(
            BasicBlock(4*k, 8*k),
            BasicBlock(8*k, 8*k)
        )
        
        #self.aspp = ASPP(8*k, 8*k, [1, 6, 12, 18])
        self.aspp = nn.Sequential(
            BasicBlock(8*k, 8*k, dilation=2),
            BasicBlock(8*k, 8*k, dilation=4),
            BasicBlock(8*k, 8*k, dilation=8)
        )
        self.fc = nn.Conv3d(8*k, num_classes, kernel_size=1, bias=False)
        
    def forward(self, x):
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        
        return x
    
class DAResNet3d(nn.Module):
    
    def __init__(self, input_channels=1, num_classes = 2, k=16, deep_supervision=False, heat_map=False):
        
        super(DAResNet3d, self).__init__()
        self.deep_supervision = deep_supervision
        self.heat_map = heat_map
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(input_channels, k, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(k, k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', norm(k)),
            ('relu2', nn.ReLU(inplace=True))]
        ))
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock,   k, 3, kernel_size=(3,3,3), stride=1)
        self.layer2 = self._make_layer(BasicBlock, 2*k, 4, kernel_size=(1,3,3), stride=2)
        self.layer3 = self._make_layer(BasicBlock, 4*k, 6, kernel_size=(1,3,3), stride=(1,2,2))
        self.layer4 = self._make_layer(BasicBlock, 8*k, 3, kernel_size=(1,3,3), stride=(1,2,2))
        
        self.class4 = DANetHead(8*k, 8*k)
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(8*k, 8*k, kernel_size=(1,2,2), stride=(1,2,2)),
            norm(8*k),
            nn.ReLU(inplace=False)
        )
        self.class3 = nn.Sequential(
            CBR(4*k+8*k, 4*k, (1,3,3))
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(4*k, 4*k, kernel_size=(1,2,2), stride=(1,2,2)),
            norm(4*k),
            nn.ReLU(inplace=False)
        )
        self.class2 = nn.Sequential(
            CBR(2*k+4*k, 2*k, (3,3,3)),
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(2*k, 2*k, kernel_size=2, stride=2),
            norm(2*k),
            nn.ReLU(inplace=False)
        )
        
        self.class1 = nn.Sequential(
            CBR(k+2*k, 2*k),
            nn.Conv3d(2*k, num_classes, kernel_size=1, bias=False),
        )
        
        if self.deep_supervision:
            self.seg4 = nn.Conv3d(8*k, num_classes, kernel_size=1, bias=False)
            #self.seg3 = nn.Conv3d(4*k, num_classes, kernel_size=1, bias=False)
            self.seg2 = nn.Conv3d(2*k, num_classes, kernel_size=1, bias=False)
        
        if self.heat_map:
            self.heatmap = nn.Sequential(
                CBR(k+2*k, 2*k),
                nn.Conv3d(2*k, 1, kernel_size=1, bias=False),
            )
        
    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.class4(self.layer4(x3))
        
        if self.training and self.deep_supervision:
            s4 = self.seg4(x4)
            s4 = F.interpolate(s4, x_size[2:], mode='trilinear', align_corners=True)
        
        x = self.class3(torch.cat([self.up3(x4), x3], 1))
        
        #if self.training and self.deep_supervision:
        #    s3 = self.seg3(x)
        #    s3 = F.interpolate(s3, x_size[2:], mode='trilinear', align_corners=True)
        
        x = self.class2(torch.cat([self.up2(x), x2], 1))
        
        if self.training and self.deep_supervision:
            s2 = self.seg2(x)
            s2 = F.interpolate(s2, x_size[2:], mode='trilinear', align_corners=True)
        
        x = torch.cat([self.up1(x), x1], 1)
        if self.heat_map:
            hp = self.heatmap(x)
            hp = F.interpolate(hp, x_size[2:], mode='trilinear', align_corners=True)
        
        x = self.class1(x)
        x = F.interpolate(x, x_size[2:], mode='trilinear', align_corners=True)
        
        
        if self.training and self.deep_supervision:
            return {'y':(x, s2, s4), 'heatmap':hp} if self.heat_map else {'y':(x, s2, s4)}
        else:
            return {'y': x, 'heatmap': hp} if self.heat_map else {'y': x}



    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

nonlinearity = partial(F.relu, inplace=True)

class DACBlock(nn.Module):
    def __init__(self, channel, kSize=(3,3,3)):
        super(DACBlock, self).__init__()
        self.dilate1 = C(channel, channel, kSize, dilation=1)
        self.dilate2 = C(channel, channel, kSize, dilation=3)
        self.dilate3 = C(channel, channel, kSize, dilation=5)
        self.conv1x1x1 = nn.Conv3d(channel, channel, kernel_size=1, dilation=1, padding=0)
        
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        
        return out
    
class SPPBlock(nn.Module):
    def __init__(self, channel):
        super(SPPBlock, self).__init__()
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=(3,3,3), stride=3)
        self.pool3 = nn.MaxPool3d(kernel_size=(5,5,5), stride=(5,5,5))
        self.pool4 = nn.MaxPool3d(kernel_size=(6,6,6), stride=(6,6,6))
        
        self.conv = nn.Conv3d(channel, 1, kernel_size=1, padding=0)
        
    def forward(self, x):
        b, c, d, h, w = x.size()
        self.layer1 = F.interpolate(self.conv(self.pool1(x)), (d, h, w), mode='trilinear', align_corners=True)
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), (d, h, w), mode='trilinear', align_corners=True)
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), (d, h, w), mode='trilinear', align_corners=True)
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), (d, h, w), mode='trilinear', align_corners=True)
        
        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)
        
        return out

class CENet(nn.Module):
    
    def __init__(self, num_classes = 2, k=16, deep_supervision=False, heat_map=False):
        
        super(CENet, self).__init__()
        self.deep_supervision = deep_supervision
        self.heat_map = heat_map
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, k, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(k, k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', norm(k)),
            ('relu2', nn.ReLU(inplace=True))]
        ))
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock,   k, 3, kernel_size=(3,3,3), stride=1)
        self.layer2 = self._make_layer(BasicBlock, 2*k, 4, kernel_size=(1,3,3), stride=2)
        self.layer3 = self._make_layer(BasicBlock, 4*k, 6, kernel_size=(1,3,3), stride=(1,2,2))
        self.layer4 = self._make_layer(BasicBlock, 8*k, 3, kernel_size=(1,3,3), stride=(1,2,2))        
        
        self.dblock = DACBlock(8*k)
        self.spp = SPPBlock(8*k)
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(8*k+4, 8*k, kernel_size=(1,2,2), stride=(1,2,2)),
            norm(8*k),
            nn.ReLU(inplace=True)
        )
        
        self.class3 = nn.Sequential(
            CBR(4*k+8*k, 4*k, (1,3,3))
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(4*k, 4*k, kernel_size=(1,2,2), stride=(1,2,2)),
            norm(4*k),
            nn.ReLU(inplace=True)
        )
        self.class2 = nn.Sequential(
            CBR(2*k+4*k, 2*k, (3,3,3)),
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(2*k, 2*k, kernel_size=2, stride=2),
            norm(2*k),
            nn.ReLU(inplace=True)
        )
        
        self.class1 = nn.Sequential(
            CBR(k+2*k, 2*k),
            #nn.Conv3d(2*k, num_classes, kernel_size=1, bias=False),
            nn.ConvTranspose3d(2*k, k, kernel_size=2, stride=2),
            norm(k),
            nn.ReLU(inplace=True),
            nn.Conv3d(k, num_classes, kernel_size=3, padding=1, bias=False)
        )  
        
        if self.deep_supervision:
            self.seg4 = nn.Conv3d(8*k+4, num_classes, kernel_size=1, bias=False)
            #self.seg3 = nn.Conv3d(4*k, num_classes, kernel_size=1, bias=False)
            self.seg2 = nn.Conv3d(2*k, num_classes, kernel_size=1, bias=False)
        
        if self.heat_map:
            self.heatmap = nn.Sequential(
                CBR(k+2*k, 2*k),
                nn.ConvTranspose3d(2*k, k, kernel_size=2, stride=2),
                norm(k),
                nn.ReLU(inplace=True),
                nn.Conv3d(k, 1, kernel_size=3, padding=1, bias=False)
            )
        
        
        #for m in self.modules():
        #    if isinstance(m, nn.Conv3d):
        #        n = reduce((lambda x, y: x * y), m.kernel_size) * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, nn.BatchNorm3d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()
        
    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        x4 = self.spp(self.dblock(self.layer4(x3)))
        
        if self.training and self.deep_supervision:
            s4 = self.seg4(x4)
            s4 = F.interpolate(s4, x_size[2:], mode='trilinear', align_corners=True)
        
        x = self.class3(torch.cat([self.up3(x4), x3], 1))
        
        #if self.training and self.deep_supervision:
        #    s3 = self.seg3(x)
        #    s3 = F.interpolate(s3, x_size[2:], mode='trilinear', align_corners=True)
        
        x = self.class2(torch.cat([self.up2(x), x2], 1))
        
        if self.training and self.deep_supervision:
            s2 = self.seg2(x)
            s2 = F.interpolate(s2, x_size[2:], mode='trilinear', align_corners=True)
        
        x = torch.cat([self.up1(x), x1], 1)
        
        if self.heat_map:
            hp = self.heatmap(x)
        
        x = self.class1(x)
        
        if self.training and self.deep_supervision:
            return {'y':(x, s2, s4), 'heatmap':hp} if self.heat_map else {'y':(x, s2, s4)}
        else:
            return {'y': x, 'heatmap': hp} if self.heat_map else {'y': x}
        

    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)
    

class MIXNet(nn.Module):
    
    def __init__(self, num_classes = 2, k=16, deep_supervision=False, heat_map=False):
        
        super(MIXNet, self).__init__()
        self.deep_supervision = deep_supervision
        self.heat_map = heat_map
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, k, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(k, k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', norm(k)),
            ('relu2', nn.ReLU(inplace=True))]
        ))
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock,   k, 3, kernel_size=(3,3,3), stride=1)
        self.layer2 = self._make_layer(BasicBlock, 2*k, 4, kernel_size=(1,3,3), stride=2)
        self.layer3 = self._make_layer(BasicBlock, 4*k, 6, kernel_size=(1,3,3), stride=(1,2,2))
        self.layer4 = self._make_layer(BasicBlock, 8*k, 3, kernel_size=(1,3,3), stride=(1,2,2))        
        
        self.da_block = nn.Sequential(
            nn.Conv3d(8*k, 4*k, kernel_size=1, bias=False),
            DANetHead(4*k, 4*k)
        )
        
        self.ce_block = nn.Sequential(
            nn.Conv3d(8*k, 4*k, kernel_size=1, bias=False),
            DACBlock(4*k),
            SPPBlock(4*k)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(8*k+4, 8*k, kernel_size=(1,2,2), stride=(1,2,2)),
            norm(8*k),
            nn.ReLU(inplace=True)
        )
        
        self.class3 = nn.Sequential(
            CBR(4*k+8*k, 4*k, (1,3,3))
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(4*k, 4*k, kernel_size=(1,2,2), stride=(1,2,2)),
            norm(4*k),
            nn.ReLU(inplace=True)
        )
        self.class2 = nn.Sequential(
            CBR(2*k+4*k, 2*k, (3,3,3)),
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(2*k, 2*k, kernel_size=2, stride=2),
            norm(2*k),
            nn.ReLU(inplace=True)
        )
        
        self.up0 = nn.Sequential(
            CBR(k+2*k, 2*k),
            nn.ConvTranspose3d(2*k, k, kernel_size=2, stride=2),
            norm(k),
            nn.ReLU(inplace=True)
        )
        
        self.class0 = nn.Conv3d(k, num_classes, kernel_size=3, padding=1, bias=False) 
        
        if self.deep_supervision:
            self.seg4 = nn.Conv3d(8*k+4, num_classes, kernel_size=1, bias=False)
            #self.seg3 = nn.Conv3d(4*k, num_classes, kernel_size=1, bias=False)
            self.seg2 = nn.Conv3d(2*k, num_classes, kernel_size=1, bias=False)
        
        if self.heat_map:
            self.heatmap = nn.Conv3d(k, 1, kernel_size=3, padding=1, bias=False)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = reduce((lambda x, y: x * y), m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        x4 = self.layer4(x3)
        x4_0 = self.da_block(x4)
        x4_1 = self.ce_block(x4)
        x4 = torch.cat([x4_0, x4_1], 1)
        
        if self.training and self.deep_supervision:
            s4 = self.seg4(x4)
            s4 = F.interpolate(s4, x_size[2:], mode='trilinear', align_corners=True)
        
        x = self.class3(torch.cat([self.up3(x4), x3], 1))
        
        #if self.training and self.deep_supervision:
        #    s3 = self.seg3(x)
        #    s3 = F.interpolate(s3, x_size[2:], mode='trilinear', align_corners=True)
        
        x = self.class2(torch.cat([self.up2(x), x2], 1))
        
        if self.training and self.deep_supervision:
            s2 = self.seg2(x)
            s2 = F.interpolate(s2, x_size[2:], mode='trilinear', align_corners=True)
        
        x = torch.cat([self.up1(x), x1], 1)
        x = self.up0(x)
        
        if self.heat_map:
            hp = self.heatmap(x)
        
        x = self.class0(x)
        
        if self.training and self.deep_supervision:
            return {'y':(x, s2, s4), 'heatmap':hp} if self.heat_map else {'y':(x, s2, s4)}
        else:
            return {'y': x, 'heatmap': hp} if self.heat_map else {'y': x}
        

    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)
        
