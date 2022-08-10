from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce

from .resunet import norm, CBR, BasicBlock, DownSample, DANetHead, DACBlock, SPPBlock 


class DAResUNet(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, psp=True):
        
        super(DAResUNet, self).__init__()
        
        self.layer0 = CBR(1, k, 7, 1)
        self.class0 = nn.Sequential(
            BasicBlock(k+2*k, 2*k),
            nn.Conv3d(2*k, segClasses, kernel_size=1, bias=False)
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
        output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=True)
        #output = self.up3(output)
        output = self.class2(torch.cat([output2, output], 1))
        #output = self.up2(output)
        output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=True)
        output = self.class1(torch.cat([output1, output], 1))
        #output = self.up1(output)
        output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=True)
        output = self.class0(torch.cat([output0, output], 1))
        
        return {'y':output}

class DAResUNet2(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, psp=True):
        
        super(DAResUNet2, self).__init__()
        
        self.layer0 =nn.Sequential(
            CBR(1, k, 3, 1),
            CBR(k, 2*k, 3, 1),
            CBR(2*k, 2*k, 3, 1)
        )
        
        self.class0 = nn.Sequential(
            BasicBlock(2*k+2*k, 2*k),
            nn.Conv3d(2*k, segClasses, kernel_size=1, bias=False)
        )
        
        self.pool1 = DownSample(2*k, 4*k, 'conv')
        self.layer1 = nn.Sequential(
            BasicBlock(4*k, 4*k),
            BasicBlock(4*k, 4*k)
        )
        
        self.class1 = nn.Sequential(
            BasicBlock(4*k+4*k, 4*k)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(4*k, 2*k, kernel_size=2, stride=2),
            norm(2*k),
            nn.ReLU(inplace=False)
        )
        
        self.pool2 = DownSample(4*k, 8*k, 'conv')
        self.layer2 = nn.Sequential(
            BasicBlock(8*k, 8*k),
            BasicBlock(8*k, 8*k)
        )
        
        self.class2 = nn.Sequential(
            BasicBlock(8*k+8*k, 8*k)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(8*k, 4*k, kernel_size=2, stride=2),
            norm(4*k),
            nn.ReLU(inplace=False)
        )
        
        
        self.pool3 = DownSample(8*k, 16*k, 'conv')
        self.layer3 = nn.Sequential(
            BasicBlock(16*k, 16*k, dilation=1),
            BasicBlock(16*k, 16*k, dilation=2),
            BasicBlock(16*k, 16*k, dilation=4)
        )
        
        self.class3 = DANetHead(16*k, 16*k)
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(16*k, 8*k, kernel_size=2, stride=2),
            norm(8*k),
            nn.ReLU(inplace=False)
        )
        
        self._init_weight()
        
    def  _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
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
        output = self.class2(torch.cat([output2, self.up3(output)], 1))
        output = self.class1(torch.cat([output1,  self.up2(output)], 1))
        output = self.class0(torch.cat([output0, self.up1(output)], 1))
        
        return {'y':output}
    
class DAResNet34(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, psp=True):
        
        super(DAResNet34, self).__init__()
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, k, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(k, k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', norm(k)),
            ('relu2', nn.ReLU(inplace=True))]
        ))
        self.inplanes = k
        self.layer1 = self._make_layer(BasicBlock, k, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 2*k, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 4*k, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 8*k, 3, stride=2)
        
        self.class4 = DANetHead(8*k, 8*k)
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(8*k, 8*k, kernel_size=2, stride=2),
            norm(8*k),
            nn.ReLU(inplace=False)
        )
        self.class3 = nn.Sequential(
            BasicBlock(4*k+8*k, 8*k),
            CBR(8*k, 4*k)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(4*k, 4*k, kernel_size=2, stride=2),
            norm(4*k),
            nn.ReLU(inplace=False)
        )
        self.class2 = nn.Sequential(
            BasicBlock(2*k+4*k, 4*k),
            CBR(4*k, 2*k)
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(2*k, 2*k, kernel_size=2, stride=2),
            norm(2*k),
            nn.ReLU(inplace=False)
        )
        
        self.class1 = nn.Sequential(
            BasicBlock(k+2*k, 2*k),
            nn.ConvTranspose3d(2*k, k, kernel_size=2, stride=2),
            norm(k),
            nn.ReLU(inplace=True),
            nn.Conv3d(k, segClasses, kernel_size=3, padding=1, bias=False),
        )
        
    def forward(self, x):
        x_size = x.size()
        
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.class4(self.layer4(x3))

        x = self.class3(torch.cat([self.up3(x4), x3], 1))
        x = self.class2(torch.cat([self.up2(x), x2], 1))
        x = self.class1(torch.cat([self.up1(x), x1], 1))
        
        return {'y': x}



    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class CENet(nn.Module):
    
    def __init__(self, segClasses = 2, k=16, deep_supervision=False):
        
        super(CENet, self).__init__()
        self.deep_supervision = deep_supervision
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(k, 2*k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', norm(2*k)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv3d(2*k, 2*k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', norm(2*k)),
            ('relu3', nn.ReLU(inplace=True))]
        ))
        
        self.inplanes = 2*k
        self.layer1 = self._make_layer(BasicBlock, 2*k, 4, kernel_size=(3,3,3), stride=2)
        self.layer2 = self._make_layer(BasicBlock, 4*k, 5, kernel_size=(3,3,3), stride=2)
        self.layer3 = self._make_layer(BasicBlock, 8*k, 7, kernel_size=(3,3,3), stride=2)
        
        self.dblock = DACBlock(8*k)
        self.spp = SPPBlock(8*k)
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(8*k+4, 8*k, kernel_size=(2,2,2), stride=(2,2,2)),
            norm(8*k),
            nn.ReLU(inplace=True)
        )
        
        self.class3 = nn.Sequential(
            CBR(4*k+8*k, 4*k)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(4*k, 4*k, kernel_size=(2,2,2), stride=(2,2,2)),
            norm(4*k),
            nn.ReLU(inplace=True)
        )
        
        self.class2 = nn.Sequential(
            CBR(2*k+4*k, 2*k),
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(2*k, 2*k, kernel_size=2, stride=2),
            norm(2*k),
            nn.ReLU(inplace=True)
        )
        
        self.class1 = nn.Sequential(
            CBR(2*k+2*k, 2*k),
            nn.Conv3d(2*k, segClasses, kernel_size=3, padding=1, bias=False)
        )
        
        if self.deep_supervision:
            self.seg3 = nn.Conv3d(8*k+4, segClasses, kernel_size=1, bias=False)
            self.seg2 = nn.Conv3d(4*k, segClasses, kernel_size=1, bias=False)
            self.seg1 = nn.Conv3d(2*k, segClasses, kernel_size=1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = reduce((lambda x, y: x * y), m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        
        x_size = x.size()
        
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        
        x3 = self.spp(self.dblock(self.layer3(x2)))
        
        if self.training and self.deep_supervision:
            s3 = self.seg3(x3)
            s3 = F.interpolate(s4, x_size[2:], mode='trilinear', align_corners=True)
        
        x = self.class3(torch.cat([self.up3(x3), x2], 1))
        
        if self.training and self.deep_supervision:
            s2 = self.seg2(x)
            s2 = F.interpolate(s2, x_size[2:], mode='trilinear', align_corners=True)
        
        x = self.class2(torch.cat([self.up2(x), x1], 1))
        
        if self.training and self.deep_supervision:
            s1 = self.seg1(x)
            s1 = F.interpolate(s1, x_size[2:], mode='trilinear', align_corners=True)
        
        x = self.class1(torch.cat([self.up1(x), x0], 1))
        
        if self.training and self.deep_supervision:
            return {'y': (x, s1, s2, s3)}
        else:
            return {'y': x}
    
    def _make_layer(self, block, planes, blocks, kernel_size=(3,3,3), stride=1, dilation=1):

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)
    