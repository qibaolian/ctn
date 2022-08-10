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

from tasks.aneurysm.nets.resunet import norm, CB, CBR, BasicBlock, DownSample, DANetHead, DACBlock, SPPBlock, PSPModule
from tasks.aneurysm.nets.vessel_net import Decoder, cSE, sSE

class SimpleCoronaryNet(nn.Module):
    
    def __init__(self, segClasses = 2, k=32):
        
        super(SimpleCoronaryNet, self).__init__()
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True))]))
        
        self.layer1 = nn.Sequential(
            CBR(k,   k),
            CBR(k,   k),
            CBR(k,   k, dilation=2),
            CBR(k,   k, dilation=4),
            CBR(k,   2*k),
            CBR(2*k, 2*k, kSize=1),
            nn.Conv3d(2*k, segClasses, kernel_size=1, bias=False),
        )
    
    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        return {'y': x}

class SE(nn.Module):
    
    def __init__(self, channels):
        super(SE, self).__init__()        
        self.cse = cSE(channels)
        self.sse = sSE(channels)
    
    def forward(self, x):
        g1 = self.sse(x)
        g2 = self.cse(x)
        
        return g1*x + g2*x

class SimpleCoronaryNet2(nn.Module):
    
    def __init__(self, segClasses = 2, k=32, se=True):
        
        super(SimpleCoronaryNet2, self).__init__()
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, k, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True))]))
        
        
        self.layer1 = nn.Sequential(
            CBR(k, k),
            CBR(k, k, dilation=2)
        )
        
        self.layer2 = nn.Sequential(
            CBR(k, 2*k),
            CBR(2*k, 2*k, dilation=2),
            CBR(2*k, 2*k, dilation=4)
        )
        
        self.se = se
        if se:
            self.se0 = SE(k)
            self.se1 = SE(k)
            self.se2 = SE(2*k)
        
        self.classify = nn.Sequential(
            CBR(4*k, 2*k, kSize=1),
            nn.Conv3d(2*k, segClasses, kernel_size=1, bias=False),
        )
        
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        
    def forward(self, x):
        
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        
        if self.se:
            x0 = self.se0(x0)
            x1 = self.se1(x1)
            x2 = self.se2(x2)
        
        x0 = self.dropout(x0)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        
        y = self.classify(torch.cat([x0, x1, x2], 1))
        
        return {'y': y}
    
class CoronaryUNet(nn.Module):
    
    def __init__(self, segClasses = 2, k=32, psp=False):
        
        super(CoronaryUNet, self).__init__()
        
        self.psp = psp
        
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
        
        if self.psp:
            #sizes=((1,1,1), (2,2,2), (3, 3, 3), (6, 6, 6))
            sizes=((1,1,1), (2,2,2), (4, 4, 4))
            self.psp_module = PSPModule(4*k, 4*k, sizes)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear')
    
    def forward(self, x):
        
        output0 = self.layer0(x)
        output1_0 = self.pool1(output0)
        output1 = self.layer1(output1_0)
        
        output2_0 = self.pool2(output1)
        output2 = self.layer2(output2_0)
        
        if self.psp:
            output = self.psp_module(output2)
        else:
            output = output2
        
        output = self.up2(output)
        output = self.class1(torch.cat([output1, output], 1))
        output = self.up1(output)
        output = self.class0(torch.cat([output0, output], 1))
        
        return {'y': output }