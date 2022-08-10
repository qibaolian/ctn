

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from collections import OrderedDict
import math
from functools import partial

from models.tools.module_helper import BN, CONV, DECONV, MAXPOOL, AVGPOOL, UPSAMPLE, DROPOUT, ADAPTIVEAVGPOOL


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, residual=True, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = CONV(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = BN(planes)
        self.conv2 = CONV(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BN(planes)
        self.downsample = None
        if residual and (stride != 1 or inplanes != planes):
            self.downsample = nn.Sequential(
                CONV(inplanes, planes, 1, stride, 0, bias=False),
                BN(planes)
            )
        self.residual = residual
        self.droprate = dropRate
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.droprate > 0.0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        if self.residual:
            out += residual
            
        out = self.relu(out)
        
        return out

class _TransitionDown(nn.Module):
    def __init__(self, num_input_features, num_output_features, drop_rate, pool='max'):
        super(_TransitionDown, self).__init__()
        
        if pool == 'avg' or pool == 'max':
            if num_input_features != num_output_features:
                self.pool = nn.Sequential(AVGPOOL(kernel_size=2, stride=2) if pool == 'avg' else \
                                          MAXPOOL(kernel_size=2, stride=2),
                                          CONV(num_input_features, num_output_features, 
                                               kernel_size=1, stride=1, bias=False),
                                          BN(num_output_features),
                                          nn.ReLU(inplace=True),
                                         )
            else:
                self.pool = AVGPOOL(kernel_size=2, stride=2) if pool == 'avg' else \
                            MAXPOOL(kernel_size=2, stride=2)
        else:
            self.pool = nn.Sequential(
                                      CONV(num_input_features, num_output_features, kernel_size=3, 
                                                stride=2, padding=1, bias=False),
                                      BN(num_output_features),
                                      nn.ReLU(inplace=True),
                                     )
        self.drop_rate = drop_rate
    
    def forward(self, x):
        
        if self.drop_rate > 0:
            x = F.dropout(feature, p=self.drop_rate, training=self.training)
        
        return self.pool(x)

class _TransitionUp(nn.Module):
    
    def __init__(self, num_input_features, num_output_features, drop_rate, pool='conv'):
        
        super(_TransitionUp, self).__init__()
        self.drop_rate = drop_rate
        
        if pool == 'conv':
            self.up = nn.Sequential(
                                      #nn.ConvTranspose3d(num_input_features, num_output_features, 
                                      #                   kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)),
                                      DECONV(num_input_features, num_output_features, kernel_size=2, stride=2, padding=0,
                                             output_padding=0, bias=False),
                                      BN(num_output_features),
                                      nn.ReLU(inplace=True)
                                     )
        else:
            
            if num_input_features != num_output_features:
                self.up = nn.Sequential(
                                        UPSAMPLE(scale_factor=2),
                                        CONV(num_input_features, num_output_features, 
                                             kernel_size=1, stride=1, bias=False),
                                        BN(num_output_features),
                                        nn.ReLU(inplace=True),
                                       )
            else:
                self.up = UPSAMPLE(scale_factor=2)
    
    def forward(self, x, skip):
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        
        x = self.up(x)        
        out = torch.cat([x, skip], 1)
        return out
            

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2, is_cse=True, is_sse=True, is_bn=False):
        super(SELayer, self).__init__()
        self.is_cse = is_cse
        self.is_sse = is_sse
        
        self.avg_pool = ADAPTIVEAVGPOOL(1)
        self.fc = nn.Sequential(
                      nn.Linear(channel, channel//reduction),
                      nn.ReLU(inplace=True),
                      nn.Linear(chanenl//reduction, channel),
                      nn.Sigmoid()
                  )
        self.sse1 = CONV(channel, 1, 1)
        self.sse2 = nn.Sigmoid()
        self.bn = BN(channel)
    
    def forward(self, x):
        
        b, c = x.size()[:2]
        y_c = self.avg_pool(x).view(b, c)
        
        #y_c = self.fc(y_c).view(b, c, 1, 1, 1)
        y_c = self.fc(y_c).view([b, c] + [1]*len(x.size()[2:]))
        
        if self.is_bn:
            out_c = self.bn(x*y_c)
        else:
            out_c = x*y_c
        
        y_s = self.sse2(self.sse1(x))
        if self.is_bn:
            out_s = self.bn(x*y_s)
        else:
            out_s = x * y_s
        
        if self.is_cse and not self.is_sse:
            return out_c
        elif self.is_sse and not self.is_cse:
            return out_s
        else:
            return out_c + out_s
        
        
class UNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=2, drop_rate=0, residual=False,
                 down_blocks=(2,2,2,2), up_blocks=(2,2,2,2), bottleneck_blocks=4,
                 filters=[64, 128, 256, 512], bottleneck_channel=1024):
        
        super(UNet, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        
        #self.in_conv = nn.Sequential(
        #                   CONV(input_channel, first_conv_channel, kernel_size=3, stride=1, padding=1, bias=False),
        #                   BN(first_conv_channel),
        #                   nn.ReLU(inplace=False)
        #               )
        
        self.downBlocks = nn.ModuleList([])
        self.downs = nn.ModuleList([])
        self.upBlocks = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        skip_connection_channels = []
        prev_block_channels = input_channel
        for i in range(len(down_blocks)):
            self.downBlocks.append(self.make_layer(BasicBlock, prev_block_channels, filters[i],
                                                   down_blocks[i], residual))
            skip_connection_channels.insert(0, filters[i])
            self.downs.append(_TransitionDown(filters[i], filters[i], drop_rate, 'max'))
            
            prev_block_channels = filters[i]
        
        self.center = self.make_layer(BasicBlock, prev_block_channels, bottleneck_channel,
                                      bottleneck_blocks, residual)
        prev_block_channels = bottleneck_channel
        for i in range(len(up_blocks)):
            self.ups.append(_TransitionUp(prev_block_channels, prev_block_channels, drop_rate,
                                          'conv'))
            prev_block_channels += skip_connection_channels[i]
            self.upBlocks.append(self.make_layer(BasicBlock, prev_block_channels, filters[-(i+1)],
                                                 up_blocks[-(i+1)], residual))
            prev_block_channels = filters[-(i+1)]
        
        self.final_conv = CONV(prev_block_channels, num_classes, 1, 1, 0, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                n = reduce((lambda x, y: x * y), m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        out = x
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.downBlocks[i](out)
            skip_connections.append(out)
            out = self.downs[i](out)
        out = self.center(out)
        
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.ups[i](out, skip)
            del skip
            out = self.upBlocks[i](out)
            
        out = self.final_conv(out)
        return out
    
    def make_layer(self, block, inplanes, planes, block_num, residual, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride, residual))
        for i in range(1, block_num):
            layers.append(block(planes, planes, 1, residual))
        return nn.Sequential(*layers)
    
class SEUNet(UNet):
    def __init__(self, input_channel=3, num_classes=2, drop_rate=0, residual=False,
                 down_blocks=(2,2,2,2), up_blocks=(2,2,2,2), bottleneck_blocks=4,
                 filters=[64, 128, 256, 512], bottleneck_channel=1024):
        super(SEUNet, self).__init__(input_channel, num_classes, drop_rate, redisual,
                                     down_blocks, up_blocks, bottleneck_blocks,
                                     filters, bottleneck_channel)
        
        self.selayersDwon = nn.ModuleList([])
        self.seLayersUp = nn.ModuleList([])
        
        for i in range(len(down_blocks)):
            self.seLayersDown.append(SELayer(filters[i]))
        
        for i in range(len(up_blocks)):
            self.seLayersUp.append(SELayer(filters[-(i+1)]))
        
        self.center_se = SELayer(bottleneck_channels)
   
    def forward(self, x):
        
        out = x
        
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.downBlocks[i](out)
            out = self.seLayersDown[i](out)
            skip_connections.append(out)
            out = self.downs[i](out)
        
        out = self.center(out)
        out = self.center_se(out)
        
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.ups[i](out, skip)
            out = self.upBlocks[i](out)
            out = self.seLayersUp[i](out)
        
        out = self.final_conv(out)
        return out

def UNet3_32(input_channel, num_classes):
    
    return UNet(input_channel=input_channel, num_classes=num_classes, down_blocks=(1,1,1),
                up_blocks=(1,1,1), bottleneck_blocks=2, filters = [32, 64, 128],
                bottleneck_channel = 256)

def UNet4_32(input_channel, num_classes):
    
    return UNet(input_channel=input_channel, num_classes=num_classes, down_blocks=(1,1,1,1),
                up_blocks=(1,1,1,1), bottleneck_blocks=2, filters = [32, 64, 128, 256],
                bottleneck_channel = 512)

def UNet3_64(input_channel, num_classes):
    
    return UNet(input_channel=input_channel, num_classes=num_classes, down_blocks=(1,1,1),
                up_blocks=(1,1,1), bottleneck_blocks=2, filters = [64, 128, 256],
                bottleneck_channel = 512)

def UNet4_64(input_channel, num_classes):
    
    return UNet(input_channel=input_channel, num_classes=num_classes, down_blocks=(1,1,1,1),
                up_blocks=(1,1,1,1), bottleneck_blocks=2, filters = [64, 128, 256, 512],
                bottleneck_channel = 1024)

def SEUNet3_32(input_channel, num_classes):
    
    return SEUNet(input_channel=input_channel, num_classes=num_classes, down_blocks=(1,1,1),
                  up_blocks=(1,1,1), bottleneck_blocks=2, filters = [32, 64, 128],
                  bottleneck_channel = 256)

def SEUNet4_32(input_channel, num_classes):
    
    return SEUNet(input_channel=input_channel, num_classes=num_classes, down_blocks=(1,1,1,1),
                  up_blocks=(1,1,1,1), bottleneck_blocks=2, filters = [32, 64, 128, 256],
                  bottleneck_channel = 512)

def SEUNet3_64(input_channel, num_classes):
    
    return SEUNet(input_chanenl=input_channel, num_classes=num_classes, down_blocks=(1,1,1),
                  up_blocks=(1,1,1), bottleneck_blocks=2, filters = [64, 128, 256],
                  bottleneck_channel = 512)

def SEUNet4_64(input_channel, num_classes):
    
    return SEUNet(input_chanenl=input_channel, num_classes=num_classes, down_blocks=(1,1,1,1),
                  up_blocks=(1,1,1,1), bottleneck_blocks=2, filters = [64, 128, 256, 512],
                  bottleneck_channel = 1024)

UNET_DICT = {
    'unet3_32': UNet3_32,
    'unet3_64': UNet3_64,
    'unet4_32': UNet4_32,
    'unet4_64': UNet4_64,
    'seunet3_32': SEUNet3_32,
    'seunet3_64': SEUNet3_64,
    'seunet4_32': SEUNet4_32,
    'seunet4_64': SEUNet4_64,
}

class UNET(object):
    
    def __init__(self, arch):
        self.arch = arch
    
    def __call__(self, input_channel, num_classes):
        
        model = None
        kwargs = {'input_channel': input_channel,
                  'num_classes': num_classes}
        if self.arch in UNET_DICT:
            return UNET_DICT[self.arch](**kwargs)
        else:
            raise NameError('unknown unet type %s' % self.arch)
        
        return model
                                
                              