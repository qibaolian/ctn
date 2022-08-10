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

from tasks.aneurysm.nets.resunet import BasicBlock, norm, CB, CBR, DANetHead
from .squeeze_and_excitation_3d import ChannelSELayer3D, SpatialSELayer3D
from tasks.aneurysm.nets.vessel_net import Decoder
class WBlock(nn.Module):
    
    def __init__(self, nIn, nOut):
        super(WBlock, self).__init__()
        
        self.conv = CBR(nIn, nOut, kSize=1, stride=1)
        self.spatial_gate = SpatialSELayer3D(nOut)
        self.channel_gate = ChannelSELayer3D(nOut)
    
    def forward(self, x):
        x = self.conv(x)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        
        return g1*x + g2*x

class Decoder2(nn.Module):
    
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder2, self).__init__()
        self.conv1 = CBR(in_channels, channels)
        self.conv2 = CBR(channels, out_channels)
        self.spatial_gate = SpatialSELayer3D(out_channels)
        self.channel_gate = ChannelSELayer3D(out_channels)
        
        self.reduce = CB(in_channels, out_channels, 1)
    
    def forward(self, x, e=None):
        if e is not None:
            x = torch.cat([e,
                   F.interpolate(x, e.size()[2:], mode='trilinear', align_corners=True)], 1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        
        r = self.reduce(x)
        x = self.conv1(x)
        x = self.conv2(x)
        
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        
        x = g1 * x + g2 * x
        
        return F.relu(x+r)

class DASEResNet18_HP(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, input_channels=1, attention=True):
        
        super(DASEResNet18_HP, self).__init__()
        self.attention = attention
        
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
        
        if attention:
            self.class4 = DANetHead(8*k, 8*k)
        
        self.class3 = Decoder(8*k+4*k, 4*k, k)
        self.class2 = Decoder(2*k+k, 2*k, k)
        self.class1 = Decoder(  k+k, 2*k, k)
        self.class0 = nn.Sequential(
            nn.Conv3d(4*k, 2*k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv3d(2*k, segClasses, kernel_size=1, bias=False)
        )
        
        self.class3_hp = Decoder(8*k+4*k, 4*k, k)
        self.class2_hp = Decoder(2*k+k, 2*k, k)
        self.class1_hp = Decoder(  k+k, 2*k, k)
        self.class0_hp = nn.Sequential(
            nn.Conv3d(4*k, 2*k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv3d(2*k, 1, kernel_size=1, bias=False)
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
            d4 = self.class4(self.layer4(e3))
        else:
            d4 = self.layer4(e3)
        
        #############segmentation############
        d3 = self.class3(d4, e3)
        d2 = self.class2(d3, e2)
        d1 = self.class1(d2, e1)
        feat = torch.cat([
                e0,
                d1,
                F.interpolate(d2, e0.size()[2:], mode='trilinear', align_corners=True),
                F.interpolate(d3, e0.size()[2:], mode='trilinear', align_corners=True)], 1)
        seg = self.class0(feat)
        seg = F.interpolate(seg, x_size[2:], mode='trilinear', align_corners=True)
        
        #############heatmap################
        d3 = self.class3_hp(d4, e3)
        d2 = self.class2_hp(d3, e2)
        d1 = self.class1_hp(d2, e1)
        feat = torch.cat([
                e0,
                d1,
                F.interpolate(d2, e0.size()[2:], mode='trilinear', align_corners=True),
                F.interpolate(d3, e0.size()[2:], mode='trilinear', align_corners=True)], 1)
        hp = self.class0_hp(feat)
        hp = F.interpolate(hp, x_size[2:], mode='trilinear', align_corners=True)
        
        return {'y': seg, 'hp': hp}

    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

class VT_Net(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, input_channels=1, add=True):
        
        super(VT_Net, self).__init__()
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(input_channels, k, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True))]
        ))
        
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock,    k, 2, kernel_size=(3,3,3), stride=1)
        self.down1 = nn.MaxPool3d(2)
        self.layer2 = self._make_layer(BasicBlock,  2*k, 2, kernel_size=(1,3,3), stride=(1,1,1))
        self.down2 = nn.MaxPool3d((1, 2, 2))
        self.layer3 = self._make_layer(BasicBlock,  4*k, 2, kernel_size=(1,3,3), stride=(1,1,1))
        self.down3 = nn.MaxPool3d((1, 2, 2))
        self.layer4 = self._make_layer(BasicBlock,  8*k, 2, kernel_size=(1,3,3), stride=(1,1,1))
        
        self.da_block = DANetHead(8*k, 8*k)
        
        self.class3 = nn.Sequential(
            CBR(4*k+8*k, 8*k, (1,3,3)),
            CBR(8*k, 4*k, (1,3,3))
        )
        
        self.class2 = nn.Sequential(
            CBR(2*k+4*k, 4*k, (1,3,3)),
            CBR(4*k, 2*k, (1,3,3))
        )
        
        self.class1 = nn.Sequential(
            CBR(k+2*k, 2*k, (3,3,3)),
            CBR(2*k, 2*k, (3,3,3))
        )
        
        self.wblock3 = WBlock(4*k, 2*k)
        self.wblock2 = WBlock(2*k, 2*k)
        self.wblock1 = WBlock(2*k, 2*k)
        self.w_add = add
        
        self.class0 = nn.Sequential(
            nn.Conv3d(4*k, 2*k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv3d(2*k, segClasses, kernel_size=1, bias=False)
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
        e2 = self.layer2(self.down1(e1))
        e3 = self.layer3(self.down2(e2))
        e4 = self.layer4(self.down3(e3))
        d4 = self.da_block(e4)
        
        d3 = self.class3(torch.cat((e3, F.interpolate(d4, e3.size()[2:], mode='trilinear', align_corners=True)), 1))
        d2 = self.class2(torch.cat((e2, F.interpolate(d3, e2.size()[2:], mode='trilinear', align_corners=True)), 1))
        d1 = self.class1(torch.cat((e1, F.interpolate(d2, e1.size()[2:], mode='trilinear', align_corners=True)), 1))
        
        w3 = self.wblock3(d3)
        w2 = self.wblock2(d2)
        w1 = self.wblock1(d1)
        
        if self.w_add:
            w2 = w2 + F.interpolate(w3, w2.size()[2:], mode='trilinear', align_corners=True)
            w1 = w1 + F.interpolate(w2, w1.size()[2:], mode='trilinear', align_corners=True)
        else:
            w2 = torch.max(w2, F.interpolate(w3, w2.size()[2:], mode='trilinear', align_corners=True))
            w1 = torch.max(w1, F.interpolate(w2, w1.size()[2:], mode='trilinear', align_corners=True))
        
        y = self.class0(torch.cat([e0, e1, w1], 1))
        y = F.interpolate(y, x_size[2:], mode='trilinear', align_corners=True)
        
        return {'y': y}
   
    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)