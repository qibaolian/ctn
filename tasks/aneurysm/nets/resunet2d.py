from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention2d import PAM_Module, CAM_Module

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if inplanes != planes or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1), 
                                  nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                  nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
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
    
class DAResNet2D(nn.Module):
    
    def __init__(self, num_classes = 2, k=16, psp=True):
        
        super(DAResNet2D, self).__init__()
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, k, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1',   nn.BatchNorm2d(k)),
            ('relu1', nn.ReLU(inplace=False)),
            ('conv2', nn.Conv2d(k, k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2',   nn.BatchNorm2d(k)),
            ('relu2', nn.ReLU(inplace=False))]
        ))
        
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock, k, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 2*k, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 4*k, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 8*k, 3, stride=2)
        
        self.class4 = DANetHead(8*k, 8*k)
        
        #self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(8*k, 8*k, kernel_size=2, stride=2),
            nn.BatchNorm2d(8*k),
            nn.ReLU(inplace=False)
        )
        self.class3 = nn.Sequential(
            conv3x3(4*k+8*k, 4*k),
            nn.BatchNorm2d(4*k),
            nn.ReLU(inplace=False)
        )
        
        #self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(4*k, 4*k, kernel_size=2, stride=2),
            nn.BatchNorm2d(4*k),
            nn.ReLU(inplace=False)
        )
        self.class2 = nn.Sequential(
            conv3x3(2*k+4*k, 2*k),
            nn.BatchNorm2d(2*k),
            nn.ReLU(inplace=False)
        )
        
        #self.up1 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(2*k, 2*k, kernel_size=2, stride=2),
            nn.BatchNorm2d(2*k),
            nn.ReLU(inplace=False)
        )
        
        self.class1 = nn.Sequential(
            conv3x3(k+2*k, 2*k),
            nn.BatchNorm2d(2*k),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(2*k, num_classes, kernel_size=1, bias=False)
        )
        
    def forward(self, x):

        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.class4(self.layer4(x3))

        x = self.class3(torch.cat([self.up3(x4), x3], 1))
        x = self.class2(torch.cat([self.up2(x), x2], 1))
        x = self.class1(torch.cat([self.up1(x), x1], 1))

        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)