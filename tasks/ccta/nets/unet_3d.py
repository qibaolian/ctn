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

from tasks.aneurysm.nets.resunet import norm, CB, CBR, DANetHead
from .squeeze_and_excitation_3d import ChannelSELayer3D, SpatialSELayer3D, ChannelSpatialSELayer3D, ProjectExciteLayer

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
    
    def __init__(self, in_ch, out_ch, block=conv_block, down_scale=2):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool3d(down_scale),
            block(in_ch, out_ch)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, conv=False, block=conv_block, up_scale=2):
        super(up, self).__init__()
        if conv:
            self.up = nn.ConvTranspose3d(in_ch-out_ch, in_ch-out_ch, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=up_scale, mode='trilinear', align_corners=True)
        
        self.conv = block(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class UNet4(nn.Module):
    def __init__(self, segClasses = 2, k=32):
        super(UNet4, self).__init__()
        self.conv0 = nn.Sequential(
            CBR(1, k, 3, 2),
            CBR(k, k),
            CBR(k, k)
        )
        
        self.down1 = down(    k, 2*k)
        self.down2 = down(2*k, 4*k, down_scale=(1,2,2))
        self.down3 = down(4*k, 8*k, down_scale=(1,2,2))
        self.down4 = down(8*k, 16*k, down_scale=(1,2,2))
        
        self.up4 = up(16*k + 8*k, 8*k, up_scale=(1,2,2))
        self.up3 = up( 8*k + 4*k, 4*k, up_scale=(1,2,2))
        self.up2 = up( 4*k + 2*k, 2*k, up_scale=(1,2,2))
        self.up1 = up( 2*k +     k, 2*k)
        
        self.out = nn.Conv3d(2*k, segClasses, 1)
        
    def forward(self, x):
        x_size = x.size()
        e0 = self.conv0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        
        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, e0)
        y = self.out(d1)        
        y = F.interpolate(y, x_size[2:], mode='trilinear', align_corners=True)
        
        return y
            
class se_conv_block(nn.Module):
    
    def __init__(self, in_ch, out_ch, SE=ProjectExciteLayer):
        super(se_conv_block, self).__init__()
        self.conv = double_conv3(in_ch, out_ch)
        self.se = SE(out_ch)
    
    def forward(self, x):
        x = self.conv(x)
        return self.se(x)
        

class SEUNet4(nn.Module):
    
    def __init__(self, segClasses = 2, k=32):
        super(SEUNet4, self).__init__()
        
        self.conv0 = nn.Sequential(
            CBR(1, k, 3, 2),
            se_conv_block(k, k),
        )
        
        self.down1 = down(    k, 2*k, block=se_conv_block)
        self.down2 = down(2*k, 4*k, block=se_conv_block, down_scale=(1,2,2))
        self.down3 = down(4*k, 8*k, block=se_conv_block, down_scale=(1,2,2))
        self.down4 = down(8*k, 16*k, block=se_conv_block, down_scale=(1,2,2))
        
        self.up4 = up(16*k + 8*k, 8*k, block=se_conv_block, up_scale=(1,2,2))
        self.up3 = up( 8*k + 4*k, 4*k, block=se_conv_block, up_scale=(1,2,2))
        self.up2 = up( 4*k + 2*k, 2*k, block=se_conv_block, up_scale=(1,2,2))
        self.up1 = up( 2*k +     k, 2*k, block=se_conv_block)
        
        self.out4 = nn.Conv3d(8*k, segClasses, 1)
        self.out3 = nn.Conv3d(4*k, segClasses, 1)
        self.out2 = nn.Conv3d(2*k, segClasses, 1)
        self.out1 = nn.Conv3d(2*k, segClasses, 1)
        
    def forward(self, x):
        x_size = x.size()
        e0 = self.conv0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        
        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, e0)
        y1 = self.out1(d1)
        
        if self.training:
            y2 = self.out2(d2)
            y3 = self.out3(d3)
            y4 = self.out4(d4)
            return {'y': (y1, y2, y3, y4)}
        else:
            y = F.interpolate(y1, x_size[2:], mode='trilinear', align_corners=True)
            return {'y': y}

class SpatialConv3d(nn.Module):
    def __init__(self, k):
        super(SpatialConv3d, self).__init__()
        
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv3d(k, k, (3, 1, 3), padding=(1, 0, 1), bias=False))
        self.message_passing.add_module('down_up', nn.Conv3d(k, k, (3, 1, 3), padding=(1, 0, 1), bias=False))
        self.message_passing.add_module('left_right', nn.Conv3d(k, k, (3, 3, 1), padding=(1, 1, 0), bias=False))
        self.message_passing.add_module('right_left', nn.Conv3d(k, k, (3, 3, 1), padding=(1, 1, 0), bias=False))
        self.message_passing.add_module('front_back', nn.Conv3d(k, k, (1, 3, 3), padding=(0, 1, 1), bias=False))
        self.message_passing.add_module('back_front', nn.Conv3d(k, k, (1, 3, 3), padding=(0, 1, 1), bias=False))
        
    def forward(self, x):
        dims = [1, 1, 2, 2, 0, 0]
        reverse = [False, True, False, True, False, True]
        
        for ms_conv, v, r in zip(self.message_passing, dims, reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        
        return x
    
    def message_passing_once(self, x, conv, dim=0, reverse=False):
        
        b, c, D, H, W = x.shape
        if dim == 0:
            slices = [x[:, :, i:i+1, :, :] for i in range(D)]
            dim = 2
        elif dim == 1:
            slices = [x[:, :, :, i:i+1, :] for i in range(H)]
            dim = 3
        else:
            slices = [x[:, :, :, :, i:i+1] for i in range(W)]
            dim = 4
        
        if reverse:
            slices = slices[::-1]
        
        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i-1])))
        
        if reverse:
                out = out[::-1]
        
        return torch.cat(out, dim=dim)

class cf_up(nn.Module):
    
    def __init__(self, in_ch, out_ch, conv=False, block=conv_block, up_scale=2, se=False, SE=ProjectExciteLayer):
        super(cf_up, self).__init__()
        if conv:
            self.up = nn.ConvTranspose3d(in_ch-out_ch, in_ch-out_ch, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=up_scale, mode='trilinear', align_corners=True)
        
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


class SECFUNet4(nn.Module):
    
    def __init__(self, segClasses = 2, k=32):
        super(SECFUNet4, self).__init__()
        
        self.conv0 = nn.Sequential(
            CBR(1, k, 3, 2),
            se_conv_block(k, k),
        )
        
        self.down1 = down(    k, 2*k, block=se_conv_block)
        self.down2 = down(2*k, 4*k, block=se_conv_block, down_scale=(1,2,2))
        self.down3 = down(4*k, 8*k, block=se_conv_block, down_scale=(1,2,2))
        self.down4 = down(8*k, 16*k, block=se_conv_block, down_scale=(1,2,2))
        
        self.up4 = cf_up(16*k + 8*k, 8*k, block=CBR, up_scale=(1,2,2), se=True)
        self.up3 = cf_up( 8*k + 4*k, 4*k, block=CBR, up_scale=(1,2,2), se=True)
        self.up2 = cf_up( 4*k + 2*k, 2*k, block=CBR, up_scale=(1,2,2), se=True)
        self.up1 = cf_up( 2*k +     k, 1*k, block=CBR, se=True)
        
        self.out4 = nn.Conv3d(8*k, segClasses, 1)
        self.out3 = nn.Conv3d(4*k, segClasses, 1)
        self.out2 = nn.Conv3d(2*k, segClasses, 1)
        self.out1 = nn.Conv3d(2*k, segClasses, 1)
        
    def forward(self, x):
        x_size = x.size()
        e0 = self.conv0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        
        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, e0)
        y1 = self.out1(torch.cat((d1, e0), 1))        
        
        if self.training:
            y0 = F.interpolate(y1, x_size[2:], mode='trilinear', align_corners=True)
            y2 = self.out2(d2)
            y3 = self.out3(d3)
            y4 = self.out4(d4)
            return {'y': (y0, y1, y2, y3, y4)}
        else:
            y = F.interpolate(y1, x_size[2:], mode='trilinear', align_corners=True)
            return {'y': y}

class SCSECFUNet4(nn.Module):
    
    def __init__(self, segClasses = 2, k=32):
        super(SCSECFUNet4, self).__init__()
        
        self.conv0 = nn.Sequential(
            CBR(1, k, 3, 2),
            se_conv_block(k, k),
        )
        
        self.down1 = down(    k, 2*k, block=se_conv_block)
        self.down2 = down(2*k, 4*k, block=se_conv_block, down_scale=(1,2,2))
        self.down3 = down(4*k, 8*k, block=se_conv_block, down_scale=(1,2,2))
        self.down4 = down(8*k, 16*k, block=se_conv_block, down_scale=(1,2,2))
        
        self.sc = SpatialConv3d(16*k)
        
        self.up4 = cf_up(16*k + 8*k, 8*k, block=CBR, up_scale=(1,2,2), se=True)
        self.up3 = cf_up( 8*k + 4*k, 4*k, block=CBR, up_scale=(1,2,2), se=True)
        self.up2 = cf_up( 4*k + 2*k, 2*k, block=CBR, up_scale=(1,2,2), se=True)
        self.up1 = cf_up( 2*k +     k, 1*k, block=CBR, se=True)
        
        self.out4 = nn.Conv3d(8*k, segClasses, 1)
        self.out3 = nn.Conv3d(4*k, segClasses, 1)
        self.out2 = nn.Conv3d(2*k, segClasses, 1)
        self.out1 = nn.Conv3d(2*k, segClasses, 1)
        
    def forward(self, x):
        x_size = x.size()
        e0 = self.conv0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        e4 = self.sc(e4)
        
        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, e0)
        y1 = self.out1(torch.cat((d1, e0), 1))  
        
        if self.training:
            y0 = F.interpolate(y1, x_size[2:], mode='trilinear', align_corners=True)
            y2 = self.out2(d2)
            y3 = self.out3(d3)
            y4 = self.out4(d4)
            return {'y': (y0, y1, y2, y3, y4)}
        else:
            y = F.interpolate(y1, x_size[2:], mode='trilinear', align_corners=True)
            return {'y': y}

class DASECFUNet4(nn.Module):
    
    def __init__(self, segClasses = 2, k=32):
        super(DASECFUNet4, self).__init__()
        
        self.conv0 = nn.Sequential(
            CBR(1, k, 3, 2),
            se_conv_block(k, k),
        )
        
        self.down1 = down(    k, 2*k, block=se_conv_block)
        self.down2 = down(2*k, 4*k, block=se_conv_block, down_scale=(1,2,2))
        self.down3 = down(4*k, 8*k, block=se_conv_block, down_scale=(1,2,2))
        self.down4 = down(8*k, 16*k, block=se_conv_block, down_scale=(1,2,2))
        
        self.da = DANetHead(16*k, 16*k)
        
        self.up4 = cf_up(16*k + 8*k, 8*k, block=CBR, up_scale=(1,2,2), se=True)
        self.up3 = cf_up( 8*k + 4*k, 4*k, block=CBR, up_scale=(1,2,2), se=True)
        self.up2 = cf_up( 4*k + 2*k, 2*k, block=CBR, up_scale=(1,2,2), se=True)
        self.up1 = cf_up( 2*k +     k, 1*k, block=CBR, se=True)
        
        self.out4 = nn.Conv3d(8*k, segClasses, 1)
        self.out3 = nn.Conv3d(4*k, segClasses, 1)
        self.out2 = nn.Conv3d(2*k, segClasses, 1)
        self.out1 = nn.Conv3d(2*k, segClasses, 1)
        
    def forward(self, x):
        x_size = x.size()
        e0 = self.conv0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        e4 = self.da(e4)
        
        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, e0)
        y1 = self.out1(torch.cat((d1, e0), 1))  
        
        if self.training:
            y0 = F.interpolate(y1, x_size[2:], mode='trilinear', align_corners=True)
            y2 = self.out2(d2)
            y3 = self.out3(d3)
            y4 = self.out4(d4)
            return {'y': (y1, y2, y3, y4)}
        else:
            y = F.interpolate(y0, y1, x_size[2:], mode='trilinear', align_corners=True)
            return {'y': y}