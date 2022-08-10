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

class Decoder(nn.Module):
    
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = CBR(in_channels, channels)
        self.conv2 = CBR(channels, out_channels)
        self.spatial_gate = SpatialSELayer3D(out_channels)
        self.channel_gate = ChannelSELayer3D(out_channels)
        
        self.reduce = CB(in_channels, out_channels, 1)
        self.dropout = nn.Dropout3d(0.05, False)
        
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
        
        return self.dropout(F.relu(x+r))
    
class Fusion(nn.Module):
    def __init__(self, k, mode='max'):
        super(Fusion, self).__init__()
        self.mode = mode
        if mode == 'concate':
            self.conv = CBR(3*k,  k)
        else:
            self.conv = nn.Sequential()
    
    def forward(self, x0, x1, x2):
        
        if self.mode == 'concate':
            return self.conv(torch.cat([x0, x1, x2], 1))
        elif self.mode == 'max':
            x = torch.max(x0, x1)
            return torch.max(x, x2)
        else:
            return (x0 + x1 + x2) / 3

class Encoder(nn.Module):
    
    def __init__(self, k=16, input_channels=1):
        
        super(Encoder, self).__init__()
        
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
    
    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        
        return (e0, e1, e2, e3, e4)
    
class MWWNet(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, attention=True, heatmap=False):
        
        super(MWWNet, self).__init__()
        self.attention = attention
        self.heatmap = heatmap
        
        self.en0 = Encoder(k)
        self.en1 = Encoder(k)
        self.en2 = Encoder(k)
        
        self.fuse0 = Fusion(k)
        self.fuse1 = Fusion(k)
        self.fuse2 = Fusion(4*k)
        self.fuse3 = Fusion(4*k)
        self.fuse4 = Fusion(8*k)
        
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
        
        if self.heatmap:
            self.class_hp = nn.Sequential(
                nn.Conv3d(4*k, 2*k, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ELU(inplace=True),
                nn.Conv3d(2*k, 1, kernel_size=1, bias=False),
                #nn.ReLU(inplace=True),
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
        
        ew0 = self.en0(x[:, 0:1, ...])
        ew1 = self.en1(x[:, 1:2, ...])
        ew2 = self.en2(x[:, 2:3, ...])
        
        e0 = self.fuse1(ew0[0], ew1[0], ew2[0])
        e1 = self.fuse1(ew0[1], ew1[1], ew2[1])
        e2 = self.fuse2(ew0[2], ew1[2], ew2[2])
        e3 = self.fuse3(ew0[3], ew1[3], ew2[3])
        e4 = self.fuse4(ew0[4], ew1[4], ew2[4])
        
        d3 = self.class3(e4, e3)
        d2 = self.class2(d3, e2)
        d1 = self.class1(d2, e1)
        
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
        