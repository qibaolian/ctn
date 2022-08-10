from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F

from .resunet import norm, CBR, DANetHead

class DownSample(nn.Module):
    def __init__(self, nIn, nOut, pool='max', scalar_factor=2):
        
        super(DownSample, self).__init__()
        
        if pool == 'conv':
            self.pool = CBR(nIn, nOut, 3, scalar_factor)
        else:
            pool = nn.MaxPool3d(kernel_size=scalar_factor, stride=scalar_factor)
            self.pool = pool
            if nIn != nOut:
                self.pool = nn.Sequential(CBR(nIn, nOut, 1, 1), pool)
    
    def forward(self, x):
        x = self.pool(x)
        return x
    
class Upsample(nn.Module):
    
    def __init__(self, nIn, nOut, pool='linear', scalar_factor=2):
        super(Upsample, self).__init__()
        
        self.pool, self.conv = None, None
        self.scalar_factor = scalar_factor
        
        if pool == 'deconv':
            self.pool = nn.ConvTranspose3d(nIn, nOut, kernel_size=scalar_factor, stride=scalar_factor)
        else:
            if nIn != nOut:
                self.conv = CBR(nIn, nOut, 1, 1)

    def forward(self, x):
        
        if self.pool is not None:
            return self.pool(x)
        
        if self.conv is not None:
            x = self.conv(x)
 
        return F.upsample(x, scale_factor=self.scalar_factor, mode='trilinear')
    
class MultiResBlock(nn.Module):
    
    def __init__(self, in_channels, filters, alpha=1.67):
        
        super(MultiResBlock, self).__init__()
        
        W = alpha * filters
        
        self.shortcut = CBR(in_channels, int(W/6)+int(W/3)+int(W/2), 1, 1)
        self.conv3x3 = CBR(in_channels, int(W/6), 3, 1)
        self.conv5x5 = CBR(int(W/6), int(W/3), 3, 1)
        self.conv7x7 = CBR(int(W/3), int(W/2), 3, 1)
        
        self.act = nn.ReLU(True)
        self.bn = norm(int(W/6)+int(W/3)+int(W/2))
        
        self.out_channels = int(W/6) + int(W/3) + int(W/2)
        
    def forward(self, x):
        
        shortcut = self.shortcut(x)
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(conv3x3)
        conv7x7 = self.conv7x7(conv5x5)
        
        out = torch.cat([conv3x3, conv5x5, conv7x7], 1)
        out = out + shortcut
        
        return self.bn(self.act(out))

class ShortResBlock(nn.Module):
    
    def __init__(self, in_channels, filters):
        
        super(ShortResBlock, self).__init__()
        
        self.shortcut = CBR(in_channels, filters, 1, 1)
        self.conv = CBR(in_channels, filters, 3, 1)
        
        self.act = nn.ReLU(True)
        self.bn = norm(filters)
        
    def forward(self, x):
        
        shortcut = self.shortcut(x)
        out = self.conv(x)
        out = out + shortcut
        
        return self.bn(self.act(out))
    
class ResPath(nn.Module):
    
    def __init__(self, in_channels, filters, blocks):
        
        super(ResPath, self).__init__()
        
        layers = []
        layers.append(ShortResBlock(in_channels, filters))
        for i in range(1, blocks):
            layers.append(ShortResBlock(filters, filters))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        
        return self.conv(x)

class UNet3(nn.Module):
    
    def __init__(self, segClasses=2, k=32 , input_channels=1):
        
        super(UNet3, self).__init__()
        
        self.block1 = nn.Sequential( CBR(1, k, 3, 1), CBR(k, k, 3, 1) )
        self.pool1 = DownSample(k, k)       
        
        self.block2 = nn.Sequential( CBR(k, k*2, 3, 1), CBR(k*2, k*2, 3, 1) )
        self.pool2 = DownSample(k*2, k*2)
        
        self.block3 = nn.Sequential( CBR(k*2, k*4, 3, 1), CBR(k*4, k*4, 3, 1) )
        self.pool3 = DownSample(k*4, k*4)
        
        self.block4 = nn.Sequential( CBR(k*4, k*8, 3, 1), CBR(k*8, k*8, 3, 1) )
        
        self.up3 = Upsample(k*8, k*4)
        self.block7 = nn.Sequential( CBR(k*8, k*4, 3, 1), CBR(k*4, k*4, 3, 1) )
        
        self.up2 = Upsample(k*4, k*2)
        self.block8 = nn.Sequential( CBR(k*4, k*2, 3, 1), CBR(k*2, k*2, 3, 1) )
        
        self.up1 = Upsample(k*2, k)
        self.block9 = nn.Sequential( CBR(k*2, k, 3, 1), CBR(k, k, 3, 1) )
        
        self.seg = nn.Conv3d(k, segClasses, kernel_size=1, bias=False)
    
    def forward(self, x):
        
        e1 = self.block1(x)
        e2 = self.block2(self.pool1(e1))
        e3 = self.block3(self.pool2(e2))
        e4 = self.block4(self.pool3(e3))

        d3 = self.block7(torch.cat([e3, self.up3(e4)], 1))
        d2 = self.block8(torch.cat([e2, self.up2(d3)], 1))
        d1 = self.block9(torch.cat([e1, self.up1(d2)], 1))
        
        return {'y': self.seg(d1)}  

class UNet4(nn.Module):
    
    def __init__(self, segClasses=2, k=32 , input_channels=1):
        
        super(UNet4, self).__init__()
        
        self.block1 = nn.Sequential( CBR(1, k, 3, 1), CBR(k, k, 3, 1) )
        self.pool1 = DownSample(k, k)       
        
        self.block2 = nn.Sequential( CBR(k, k*2, 3, 1), CBR(k*2, k*2, 3, 1) )
        self.pool2 = DownSample(k*2, k*2)
        
        self.block3 = nn.Sequential( CBR(k*2, k*4, 3, 1), CBR(k*4, k*4, 3, 1) )
        self.pool3 = DownSample(k*4, k*4)
        
        self.block4 = nn.Sequential( CBR(k*4, k*8, 3, 1), CBR(k*8, k*8, 3, 1) )
        self.pool4 = DownSample(k*8, k*8)
        
        self.block5 = nn.Sequential( CBR(k*8, k*16, 3, 1), CBR(k*16, k*16, 3, 1) )
        
        self.up4 = Upsample(k*16, k*8)
        self.block6 = nn.Sequential( CBR(k*16, k*8, 3, 1), CBR(k*8, k*8, 3, 1) )
        
        self.up3 = Upsample(k*8, k*4)
        self.block7 = nn.Sequential( CBR(k*8, k*4, 3, 1), CBR(k*4, k*4, 3, 1) )
        
        self.up2 = Upsample(k*4, k*2)
        self.block8 = nn.Sequential( CBR(k*4, k*2, 3, 1), CBR(k*2, k*2, 3, 1) )
        
        self.up1 = Upsample(k*2, k)
        self.block9 = nn.Sequential( CBR(k*2, k, 3, 1), CBR(k, k, 3, 1) )
        
        self.seg = nn.Conv3d(k, segClasses, kernel_size=1, bias=False)
    
    def forward(self, x):
        
        e1 = self.block1(x)
        e2 = self.block2(self.pool1(e1))
        e3 = self.block3(self.pool2(e2))
        e4 = self.block4(self.pool3(e3))
        e5 = self.block5(self.pool4(e4))
        
        d4 = self.block6(torch.cat([e4, self.up4(e5)], 1))
        d3 = self.block7(torch.cat([e3, self.up3(d4)], 1))
        d2 = self.block8(torch.cat([e2, self.up2(d3)], 1))
        d1 = self.block9(torch.cat([e1, self.up1(d2)], 1))
        
        return {'y': self.seg(d1)}
        
class MultiResUnet4(nn.Module):
    
    def __init__(self,  segClasses = 2, k=32, input_channels=1, attention=True):
        
        super(MultiResUnet4, self).__init__()
        self.attention = attention
        
        self.block1 = MultiResBlock(1, k)
        self.pool1 = DownSample(self.block1.out_channels, self.block1.out_channels)
        self.respath1 = ResPath(self.block1.out_channels, k, 4)
        
        self.block2 = MultiResBlock(self.block1.out_channels, k*2)
        self.pool2 = DownSample(self.block2.out_channels, self.block2.out_channels)
        self.respath2 = ResPath(self.block2.out_channels, k*2, 3)
        
        self.block3 = MultiResBlock(self.block2.out_channels, k*4)
        self.pool3 = DownSample(self.block3.out_channels, self.block3.out_channels)
        self.respath3 = ResPath(self.block3.out_channels, k*4, 2)
        
        self.block4 = MultiResBlock(self.block3.out_channels, k*4)
        self.pool4 = DownSample(self.block4.out_channels, self.block4.out_channels)
        self.respath4 = ResPath(self.block4.out_channels, k*8, 1)
        
        
        self.block5 = MultiResBlock(self.block4.out_channels, k*16)
        if self.attention:
            self.da_block = DANetHead(self.block5.out_channels, self.block5.out_channels)
        
        self.up4 = Upsample(self.block5.out_channels, k*8)
        self.block6 = MultiResBlock(k*16, k*8)
        
        self.up3 = Upsample(self.block6.out_channels, k*4)
        self.block7 = MultiResBlock(k*8, k*4)
        
        self.up2 = Upsample(self.block7.out_channels, k*2)
        self.block8 = MultiResBlock(k*4, k*2)
        
        self.up1 = Upsample(self.block8.out_channels, k)
        self.block9 = MultiResBlock(k*2, k)
        
        self.seg = nn.Conv3d(self.block9.out_channels, segClasses, kernel_size=1, bias=False)
        
    def forward(self, x):
        
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
        
        return {'y': self.seg(d1)}                            

class MultiResUnet3(nn.Module):
    
    def __init__(self,  segClasses = 2, k=32, input_channels=1, attention=True):
        
        super(MultiResUnet3, self).__init__()
        self.attention = attention
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
        if self.attention:
            self.da_block = DANetHead(self.block4.out_channels, self.block4.out_channels)
        
        self.up3 = Upsample(self.block4.out_channels, k*4)
        self.block7 = MultiResBlock(k*8, k*4)
        
        self.up2 = Upsample(self.block7.out_channels, k*2)
        self.block8 = MultiResBlock(k*4, k*2)
        
        self.up1 = Upsample(self.block8.out_channels, k)
        self.block9 = MultiResBlock(k*2, k)
        
        self.seg = nn.Conv3d(self.block9.out_channels, segClasses, kernel_size=1, bias=False)
        
    def forward(self, x):
        
        e1 = self.block1(x)
        e2 = self.block2(self.pool1(e1))
        e3 = self.block3(self.pool2(e2))
        e4 = self.block4(self.pool3(e3))
        
        if self.attention:
            e4 = self.da_block(e4)
        
        d3 = self.block7(torch.cat([self.respath3(e3), self.up3(e4)], 1))
        d2 = self.block8(torch.cat([self.respath2(e2), self.up2(d3)], 1))
        d1 = self.block9(torch.cat([self.respath1(e1), self.up1(d2)], 1))
        
        return {'y': self.seg(d1)}        

class DAMultiResUnet3_4(nn.Module):
    
    def __init__(self, segClasses = 2, k=32, input_channels=1, attention=True):
        
        super(DAMultiResUnet3_4, self).__init__()
        self.attention = attention
        
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, k, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', norm(k)),
            ('relu1', nn.ReLU(inplace=True))]
        ))
        
        self.block1 = MultiResBlock(k, k)
        self.pool1 = DownSample(self.block1.out_channels, self.block1.out_channels)
        self.respath1 = ResPath(self.block1.out_channels, k, 3)
        
        self.block2 = MultiResBlock(self.block1.out_channels, k*2)
        self.pool2 = DownSample(self.block2.out_channels, self.block2.out_channels)
        self.respath2 = ResPath(self.block2.out_channels, k*2, 2)
        
        self.block3 = MultiResBlock(self.block2.out_channels, k*4)
        self.pool3 = DownSample(self.block3.out_channels, self.block3.out_channels, scalar_factor=(1,2,2))
        self.respath3 = ResPath(self.block3.out_channels, k*4, 1)
        
        self.block4 = MultiResBlock(self.block3.out_channels, k*8)
        if self.attention:
            self.da_block = DANetHead(self.block4.out_channels, self.block4.out_channels)
        
        self.up3 = Upsample(self.block4.out_channels, k*4, scalar_factor=(1,2,2))
        self.block7 = MultiResBlock(k*8, k*4)
        
        self.up2 = Upsample(self.block7.out_channels, k*2)
        self.block8 = MultiResBlock(k*4, k*2)
        
        self.up1 = Upsample(self.block8.out_channels, k)
        self.block9 = MultiResBlock(k*2, k)
        
        self.seg = nn.Conv3d(self.block9.out_channels, segClasses, kernel_size=1, bias=False)
        
    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        
        e1 = self.block1(x)
        e2 = self.block2(self.pool1(e1))
        e3 = self.block3(self.pool2(e2))
        #e4 = self.da_block(self.block4(self.pool3(e3)))
        e4 = self.block4(self.pool3(e3))
        if self.attention:
            e4 = self.da_block(e4)
        
        d3 = self.block7(torch.cat([self.respath3(e3), self.up3(e4)], 1))
        d2 = self.block8(torch.cat([self.respath2(e2), self.up2(d3)], 1))
        d1 = self.block9(torch.cat([self.respath1(e1), self.up1(d2)], 1))
        
        return {'y': F.interpolate(self.seg(d1), x_size[2:], mode='trilinear', align_corners=True)}
        