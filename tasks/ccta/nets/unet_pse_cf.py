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

from tasks.aneurysm.nets.resunet import norm, CB, CBR


class double_conv3(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(double_conv3, self).__init__()
        self.conv1 = CBR(in_ch, out_ch)
        self.conv2 = CBR(out_ch, out_ch)
        
        #self.dropout = nn.Dropout(p=0.2, inplace=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class conv_block(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = double_conv3(in_ch, out_ch)
    
    def forward(self, x):
        return self.conv(x)

class down(nn.Module):
    
    def __init__(self, in_ch, out_ch, block=conv_block):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool3d(2),
            block(in_ch, out_ch)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, conv=False, block=conv_block):
        super(up, self).__init__()
        if conv:
            self.up = nn.ConvTranspose3d(in_ch-out_ch, in_ch-out_ch, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.conv = block(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, segClasses = 2, k=32):
        super(UNet, self).__init__()
        self.conv = conv_block(1, k)
        self.down1 = down(k, 2*k)
        self.down2 = down(2*k, 4*k)
        self.up2 = up(4*k+2*k, 2*k)
        self.up1 = up(2*k+k, k)
        self.out = nn.Conv3d(k, segClasses, 1)
    
    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.up2(x2, x1)
        x4 = self.up1(x3, x0)
        return self.out(x4)

class SE1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c = x.size()[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class SE2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(2)
        self.fc = nn.Sequential(
            nn.Conv3d(channel,channel // reduction, 1, padding=0,stride=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel// reduction,channel, 1, padding=0,stride=1,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w, z = x.size()
        y = self.avg_pool(x).view(b, c, 2, 2, 2)
        y = self.fc(y).view(b, c, 2, 2, 2)
        return x * y.repeat(1,1,h//2,w//2,z//2)
    
class SE4(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE4, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(4)
        self.fc = nn.Sequential(
            nn.Conv3d(channel,channel // reduction, 1, padding=0,stride=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel// reduction,channel, 1, padding=0,stride=1,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w, z= x.size()
        y = self.avg_pool(x).view(b, c, 4, 4, 4)
        y = self.fc(y).view(b, c, 4, 4, 4)
        return x * y.repeat(1,1,h//4,w//4, z//4)
        #y = self.avg_pool(x).view(b, c, 3, 3, 3)
        #y = self.fc(y).view(b, c, 3, 3, 3)
        #return x * y.repeat(1,1,h//3,w//3, z//3)

class SE(nn.Module):
    
    def __init__(self, out_ch):
        super(SE, self).__init__()
        self.se1=SE1(out_ch, 16)
        self.se2=SE2(out_ch, 16)
        self.se3=SE4(out_ch, 16)
        
        self.conv_reduce = CBR(3*out_ch, out_ch)
    
    def forward(self, x):
        s1 = self.se1(x)
        s2 = self.se2(x)
        s3 = self.se3(x)
        
        s = torch.cat([s1, s2, s3], 1)
        return self.conv_reduce(s)
    
class se_conv_block(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(se_conv_block, self).__init__()
        self.conv = double_conv3(in_ch, out_ch)
        self.se = SE(out_ch)
    
    def forward(self, x):
        x = self.conv(x)
        return self.se(x)
        

class SEUNet(nn.Module):
    def __init__(self, segClasses = 2, k=32):
        super(SEUNet, self).__init__()
        self.conv = se_conv_block(1, k)
        self.down1 = down(k, 2*k, block=se_conv_block)
        self.down2 = down(2*k, 4*k, block=se_conv_block)
        self.up2 = up(4*k+2*k, 2*k, block=se_conv_block)
        self.up1 = up(2*k+k, k, block=se_conv_block)
        self.out = nn.Conv3d(k, segClasses, 1)
    
    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.up2(x2, x1)
        x4 = self.up1(x3, x0)
        y = self.out(x4)
        return {'y': y}

class cf_up(nn.Module):
    
    def __init__(self, in_ch, out_ch, conv=False, block=conv_block, se=False):
        super(cf_up, self).__init__()
        if conv:
            self.up = nn.ConvTranspose3d(in_ch-out_ch, in_ch-out_ch, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.conv1 = block(in_ch, out_ch)
        self.conv_reduce = CBR(in_ch-out_ch, out_ch)
        self.conv2 = block(2*out_ch, out_ch)
        
        self.se = se
        if se:
            self.se_conv = SE(out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        xfinal1 = torch.cat([x1, x2], 1)
        xfinal1 = self.conv1(xfinal1)
        xr1 = self.conv_reduce(x1)
        xr2 = x2 - xr1
        xfinal2 = self.conv2(torch.cat([xfinal1, xr2], 1))
        if self.se:
            xfinal2 = self.se_conv(xfinal2)
        return xfinal2

class CFUNet(nn.Module):
    
    def __init__(self, segClasses = 2, k=32):
        super(CFUNet, self).__init__()
        self.conv = conv_block(1, k)
        self.down1 = down(k, 2*k)
        self.down2 = down(2*k, 4*k)
        self.up2 = cf_up(4*k+2*k, 2*k, block=CBR, se=True)
        self.up1 = cf_up(2*k+k, k, block=CBR, se=True)
        self.out = nn.Conv3d(k, segClasses, 1)
    
    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.up2(x2, x1)
        x4 = self.up1(x3, x0)
        y = self.out(x4)
        return {'y': y}
    
    

class SECFUNet(nn.Module):
    
    def __init__(self, segClasses = 2, k=32):
        super(SECFUNet, self).__init__()
        self.conv = conv_block(1, k)
        self.down1 = down(k, 2*k, block=se_conv_block)
        self.down2 = down(2*k, 4*k, block=se_conv_block)
        self.up2 = cf_up(4*k+2*k, 2*k, block=CBR, se=True)
        self.up1 = cf_up(2*k+k, k, block=CBR, se=True)
        self.out = nn.Conv3d(k, segClasses, 1)
    
    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.up2(x2, x1)
        x4 = self.up1(x3, x0)
        y = self.out(x4)
        return {'y': y}

class SECFUNet3(nn.Module):
    
    def __init__(self, segClasses = 2, k=32):
        super(SECFUNet3, self).__init__()
        self.conv = se_conv_block(1, k)
        self.down1 = down(k, 2*k, block=se_conv_block)
        self.down2 = down(2*k, 4*k, block=se_conv_block)
        self.down3 = down(4*k, 8*k, block=se_conv_block)
        self.up3 = cf_up(8*k+4*k, 4*k, block=CBR, se=True)
        self.up2 = cf_up(4*k+2*k, 2*k, block=CBR, se=True)
        self.up1 = cf_up(2*k+k, k, block=CBR, se=True)
        self.out = nn.Conv3d(k, segClasses, 1)
    
    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.up3(x3, x2)
        x5 = self.up2(x4, x1)
        x6 = self.up1(x5, x0)
        y = self.out(x6)
        return {'y': y}
                            


        