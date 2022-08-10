
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch.nn as nn
from collections import OrderedDict
from functools import reduce
from models.tools.module_helper import CONV, BN, MAXPOOL, AVGPOOL

class Block(nn.Module):
    
    expansion = 2
    
    def __init__(self, in_planes, cardinality=32, stride=1, downample=None):
        
        super(Block, self).__init__()
        group_width = cardinality * int(planes / 32)
        self.conv1 = CONV(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = BN(group_width)
        self.conv2 = CONV(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = BN(group_width)
        self.conv3 = CONV(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = BN(self.expansion*group_width)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
   
    def forward(self, x):
        
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))        
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResNetHeadA(nn.Module):
    def __init__(self, input_dim=3):
        super(ResNetHeadA, self).__init__()
        
        self.conv = nn.Sequential(OrderedDict([
                ('conv1', CONV(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', BN(64)),
                ('relu', nn.ReLU(inplace=False))]
        ))
        
        self.pool = MAXPOOL(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
    def get_num_features(self):
        return 64
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class ResNetHeadB(nn.Module):
    def __init__(self, input_dim=3):
        super(ResNetHeadB, self).__init__()
        
        self.conv = nn.Sequential(OrderedDict([
            ('conv1', CONV(input_dim, 64, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', BN(64)),
            ('relu1', nn.ReLU(inplace=False)),
            ('conv2', CONV(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', BN(64)),
            ('relu2', nn.ReLU(inplace=False)),
            ('conv3', CONV(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn3', BN(128)),
            ('relu3', nn.ReLU(inplace=False))]
        ))
        
        self.pool = MAXPOOL(kernel_size=3, stride=2, padding=1, ceil_mode=True)
    
    def get_num_features(self):
        return 128
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class ResNetHeadC(nn.Module):
    def __init__(self, input_dim=3):
        super(ResNetHeadC, self).__init__()
        
        self.conv = nn.Sequential(OrderedDict([
                ('conv1', CONV(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn1', BN(64)),
                ('relu', nn.ReLU(inplace=False)),
                ('pool', MAXPOOL(kernel_size=3, stride=2, padding=1, ceil_mode=True))]
        ))
    
    def get_num_features(self):
        return 64
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class ResNeXt(nn.Module):
    
    def __init__(self, block, layers, head= 'A', input_channel=3, cardinality=32, num_classes=1000):
        
        super(ResNeXt, self).__init__()
        
        if head == 'A':
            self.head = ResNetHeadA(input_channel)
        elif head == 'B':
            self.head = ResNetHeadB(input_channel)
        elif head == 'C':
            self.head = ResNetHeadC(input_channel)
        else:
            self.head = None
        
        self.cardinality = cardinality
        self.inplanes = self.head.get_num_features()
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgPool = AVGPOOL(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                n = reduce((lambda x, y: x * y), m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _make_layer(self, block, planes, layers, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                CONV(self.inplanes, planes * block.expansion,
                     kernel_size=1, stride=stride, bias=False),
                BN(planes * block.expansion),
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.head(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class ResNeXtModels(object):
    
    def resnet50(**kwargs):
        """Constructs a ResNeXt-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNeXt(Block, [3, 4, 6, 3], **kwargs)
        
        return model


    def resnet101(**kwargs):
        """Constructs a ResNeXt-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNeXt(Block, [3, 4, 23, 3], **kwargs)
        
        return model


    def resnet152(**kwargs):
        """Constructs a ResNeXt-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNeXt(Block, [3, 8, 36, 3], **kwargs)
        
        return model
                      
        