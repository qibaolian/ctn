from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce

from tasks.aneurysm.nets.MultiResUNet3D import DownSample, Upsample, MultiResBlock, ResPath
from tasks.ccta.nets.squeeze_and_excitation_3d import ChannelSELayer3D as CSE, SpatialSELayer3D as SSE
from tasks.aneurysm.nets.resunet import DANetHead

class Decoder(nn.Module):
    def __init__(self, nIn, nOut):
        super(Decoder, self).__init__()
        
        self.conv = CBR(nIn, nOut, 1)
        self.cse = CSE(nOut)
        self.sse = SSE(nOut)
    
    def forward(self, x):
        x = self.conv(x)
        cx = self.cse(x)
        sx = self.sse(x)
        return cx + sx


class HeadNet3(nn.Module):
    
    def __init__(self, num_classes=2, k=32):
        
        super(HeadNet3, self).__init__()
        
        self.block1 = MultiResBlock(1, k)
        self.pool1 = DownSample(self.block1.out_channels, self.block1.out_channels)
        self.respath1 = ResPath(self.block1.out_channels, k, 3)
        
        self.block2 = MultiResBlock(self.block1.out_channels, k*2)
        self.pool2 = DownSample(self.block2.out_channels, self.block2.out_channels)
        self.respath2 = ResPath(self.block2.out_channels, k*2, 2)
        
        self.block3 = MultiResBlock(self.block2.out_channels, k*4)
        self.pool3 = DownSample(self.block3.out_channels, self.block3.out_channels)
        self.respath3 = ResPath(self.block3.out_channels, k*4, 1)
        
        self.block4 = MultiResBlock(self.block3.out_channels, k*8)
        
        self.up3 = Upsample(self.block4.out_channels, k*4)
        self.block5 = MultiResBlock(k*8, k*4)
        
        self.up2 = Upsample(self.block5.out_channels, k*2)
        self.block6 = MultiResBlock(k*4, k*2)
        
        self.up1 = Upsample(self.block6.out_channels, k)
        self.block7 = MultiResBlock(k*2, k)
        
        self.seg = nn.Conv3d(self.block7.out_channels, num_classes, kernel_size=1, bias=False)
        
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
        
        e1 = self.block1(x)
        e2 = self.block2(self.pool1(e1))
        e3 = self.block3(self.pool2(e2))
        e4 = self.block4(self.pool3(e3))
        
        d = self.block5(torch.cat([self.respath3(e3), self.up3(e4)], 1))
        d = self.block6(torch.cat([self.respath2(e2), self.up2(d)], 1))
        d = self.block7(torch.cat([self.respath1(e1), self.up1(d)], 1))
        
        return {'y': self.seg(d)}
    

class SEHeadNet3(nn.Module):
    
    def __init__(self, num_classes=2, k=32, attention=True):
        super(SEHeadNet3, self).__init__()
        
        self.block1 = MultiResBlock(1, k)
        self.pool1 = DownSample(self.block1.out_channels, self.block1.out_channels)
        self.respath1 = ResPath(self.block1.out_channels, k, 3)
        
        self.block2 = MultiResBlock(self.block1.out_channels, k*2)
        self.pool2 = DownSample(self.block2.out_channels, self.block2.out_channels)
        self.respath2 = ResPath(self.block2.out_channels, k*2, 2)
        
        self.block3 = MultiResBlock(self.block2.out_channels, k*4)
        self.pool3 = DownSample(self.block3.out_channels, self.block3.out_channels)
        self.respath3 = ResPath(self.block3.out_channels, k*4, 1)
        
        self.block4 = MultiResBlock(self.block3.out_channels, k*8)
        self.attention = attention
        if attention:
            self.da_block = DANetHead(self.block4.out_channels, self.block4.out_channels)
        
        self.up3 = Upsample(self.block4.out_channels, k*4)
        self.block5 = MultiResBlock(k*8, k*4)
        
        self.up2 = Upsample(self.block5.out_channels, k*2)
        self.block6 = MultiResBlock(k*4, k*2)
        
        self.up1 = Upsample(self.block6.out_channels, k)
        self.block7 = MultiResBlock(k*2, k)
        
        self.se3 = Decoder(self.block5.out_channels, k)
        self.se2 = Decoder(self.block6.out_channels, k)
        self.se1 = Decoder(self.block7.out_channels, k)
        
        self.seg = nn.Sequential(
            nn.Conv3d(4*k, 2*k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv3d(2*k, num_classes, kernel_size=1, bias=False)
        )
    
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
        
        x_size = x.size()
        
        e1 = self.block1(x)
        e2 = self.block2(self.pool1(e1))
        e3 = self.block3(self.pool2(e2))
        e4 = self.block4(self.pool3(e3))
        if self.attention:
            e4 = self.da_block(e4)
            
        d3 = self.block5(torch.cat([self.respath3(e3), self.up3(e4)], 1))
        d2 = self.block6(torch.cat([self.respath2(e2), self.up2(d3)], 1))
        d1 = self.block7(torch.cat([self.respath1(e1), self.up1(d2)], 1))
        
        d3 = self.se3(d3)
        d2 = self.se2(d2)
        d1 = self.se1(d1)
        
        feat = torch.cat([
            e1, d1, 
            F.interpolate(d2, x_size[2:], mode='trilinear', align_corners=True),
            F.interpolate(d3, x_size[2:], mode='trilinear', align_corners=True)], 1)
        
        return {'y': self.seg(feat)}
    
class HeadNet4(nn.Module):
    
    def __init__(self, num_classes=2, k=32):
        
        super(HeadNet4, self).__init__()
        
        self.block1 = MultiResBlock(1, k)
        self.pool1 = DownSample(self.block1.out_channels, self.block1.out_channels)
        self.respath1 = ResPath(self.block1.out_channels, k, 4)
        
        self.block2 = MultiResBlock(self.block1.out_channels, k*2)
        self.pool2 = DownSample(self.block2.out_channels, self.block2.out_channels)
        self.respath2 = ResPath(self.block2.out_channels, k*2, 3)
        
        self.block3 = MultiResBlock(self.block2.out_channels, k*4)
        self.pool3 = DownSample(self.block3.out_channels, self.block3.out_channels)
        self.respath3 = ResPath(self.block3.out_channels, k*4, 2)
        
        self.block4 = MultiResBlock(self.block3.out_channels, k*8)
        self.pool4 = DownSample(self.block4.out_channels, self.block4.out_channels)
        self.respath4 = ResPath(self.block4.out_channels, k*8, 1)
        
        self.block5 = MultiResBlock(self.block4.out_channels, k*16)
        
        self.up4 = Upsample(self.block5.out_channels, k*8)
        self.block6 = MultiResBlock(k*16, k*8)
        
        self.up3 = Upsample(self.block6.out_channels, k*4)
        self.block7 = MultiResBlock(k*8, k*4)
        
        self.up2 = Upsample(self.block7.out_channels, k*2)
        self.block8 = MultiResBlock(k*4, k*2)
        
        self.up1 = Upsample(self.block8.out_channels, k)
        self.block9 = MultiResBlock(k*2, k)
        
        self.seg = nn.Conv3d(self.block9.out_channels, num_classes, kernel_size=1, bias=False)
    
    def forward(self, x):
        
        e1 = self.block1(x)
        e2 = self.block2(self.pool1(e1))
        e3 = self.block3(self.pool2(e2))
        e4 = self.block4(self.pool3(e3))
        e5 = self.block5(self.pool4(e4))
        
        d4 = self.block6(torch.cat([self.respath4(e4), self.up4(e5)], 1))
        d3 = self.block7(torch.cat([self.respath3(e3), self.up3(d4)], 1))
        d2 = self.block8(torch.cat([self.respath2(e2), self.up2(d3)], 1))
        d1 = self.block9(torch.cat([self.respath1(e1), self.up1(d2)], 1))
        
        return {'y': self.seg(d1)}
    

class SEHeadNet4(nn.Module):
    
    def __init__(self, num_classes=2, k=32, attention=True):
        super(SEHeadNet4, self).__init__()
        
        self.block1 = MultiResBlock(1, k)
        self.pool1 = DownSample(self.block1.out_channels, self.block1.out_channels)
        self.respath1 = ResPath(self.block1.out_channels, k, 4)
        
        self.block2 = MultiResBlock(self.block1.out_channels, k*2)
        self.pool2 = DownSample(self.block2.out_channels, self.block2.out_channels)
        self.respath2 = ResPath(self.block2.out_channels, k*2, 3)
        
        self.block3 = MultiResBlock(self.block2.out_channels, k*4)
        self.pool3 = DownSample(self.block3.out_channels, self.block3.out_channels)
        self.respath3 = ResPath(self.block3.out_channels, k*4, 2)
        
        self.block4 = MultiResBlock(self.block3.out_channels, k*8)
        self.pool4 = DownSample(self.block4.out_channels, self.block4.out_channels)
        self.respath4 = ResPath(self.block4.out_channels, k*8, 1)
        
        self.block5 = MultiResBlock(self.block4.out_channels, k*16)
        self.attention = attention
        
        if attention:
            self.da_block = DANetHead(self.block5.out_channels, self.block5.out_channels)
        
        self.up4 = Upsample(self.block5.out_channels, k*8)
        self.block6 = MultiResBlock(k*16, k*8)
        
        self.up3 = Upsample(self.block6.out_channels, k*4)
        self.block7 = MultiResBlock(k*8, k*4)
        
        self.up2 = Upsample(self.block7.out_channels, k*2)
        self.block8 = MultiResBlock(k*4, k*2)
        
        self.up1 = Upsample(self.block8.out_channels, k)
        self.block9 = MultiResBlock(k*2, k)
        
        self.se4 = Decoder(self.block6.out_channels, k)
        self.se3 = Decoder(self.block7.out_channels, k)
        self.se2 = Decoder(self.block8.out_channels, k)
        self.se1 = Decoder(self.block9.out_channels, k)
        
        self.seg = nn.Sequential(
            nn.Conv3d(4*k, 2*k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv3d(2*k, num_classes, kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        
        x_size = x.size()
        
        e1 = self.block1(x)
        e2 = self.block2(self.pool1(e1))
        e3 = self.block3(self.pool2(e2))
        e4 = self.block4(self.pool3(e3))
        e5 = self.block5(self.pool4(e4))
        if self.attention:
            e5 = self.da_block(e5)
        
        d4 = self.block6(torch.cat([self.respath4(e4), self.up4(e5)], 1))
        d3 = self.block7(torch.cat([self.respath3(e3), self.up3(d4)], 1))
        d2 = self.block8(torch.cat([self.respath2(e2), self.up2(d3)], 1))
        d1 = self.block9(torch.cat([self.respath1(e1), self.up1(d2)], 1))
        
        d4 = self.se4(d4)
        d3 = self.se3(d3)
        d2 = self.se2(d2)
        d1 = self.se1(d1)
        
        feat = torch.cat([
            d1, 
            F.interpolate(d2, x_size[2:], mode='trilinear', align_corners=True),
            F.interpolate(d3, x_size[2:], mode='trilinear', align_corners=True),
            F.interpolate(d4, x_size[2:], mode='trilinear', align_corners=True)], 1)
        
        return {'y': self.seg(feat)}