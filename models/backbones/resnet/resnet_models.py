from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch.nn as nn
from collections import OrderedDict
from functools import reduce
from models.tools.module_helper import CONV, BN, MAXPOOL, AVGPOOL



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return CONV(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = CONV(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = CONV(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = CONV(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

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
        
        
    
class ResNet(nn.Module):
    
    def __init__(self, block, layers, head= 'A', input_channel=3, deep_base=True, first_stride=False, num_classes=1000):
        
        super(ResNet, self).__init__()
        self.channels = []
        self.first_stride = first_stride
        #if head == 'A':
        #    self.head = ResNetHeadA(input_channel)
        #elif head == 'B':
        #    self.head = ResNetHeadB(input_channel)
        #elif head == 'C':
        #    self.head = ResNetHeadC(input_channel)
        #else:
        #    self.head = None
        #self.inplanes = self.head.get_num_features()
        
        #self.inplanes = 128 if deep_base else 64
        self.inplanes = 64
        if deep_base:
            self.conv = nn.Sequential(OrderedDict([
                ('conv1', CONV(input_channel, 64, kernel_size=3, 
                               stride=2 if first_stride else 1, padding=1, bias=False)),
                ('bn1', BN(64)),
                ('relu1', nn.ReLU(inplace=False)),
                ('conv2', CONV(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn2', BN(64)),
                ('relu2', nn.ReLU(inplace=False)),
                ('conv3', CONV(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn3', BN(64)),
                ('relu3', nn.ReLU(inplace=False))]
            ))
        else:
            self.conv = nn.Sequential(OrderedDict([
                ('conv1', CONV(input_channel, 64, kernel_size=7,
                               stride=2 if first_stride else 1, padding=3, bias=False)),
                ('bn1', BN(64)),
                ('relu', nn.ReLU(inplace=False))]
            ))
        self.channels.append(self.inplanes)
        
        self.maxpool = MAXPOOL(kernel_size=3, stride=2, padding=1)#, ceil_mode=True) 
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1) 
        self.channels.append(self.inplanes)
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.channels.append(self.inplanes)
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.channels.append(self.inplanes)
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.channels.append(self.inplanes)
        
        self.avgPool = AVGPOOL(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                n = reduce((lambda x, y: x * y), m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
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
            layers.append(block(self.inplanes, planes))
        
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

class ResNetModels(object):
    
    def resnet18(self, **kwargs):
        """Constructs a ResNet-18 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        
        return model
    
    
    def resnet34(self, **kwargs):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
        
        return model
    
    
    def resnet50(self, **kwargs):
        """Constructs a ResNet-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        
        return model
    
    
    def resnet101(self, **kwargs):
        """Constructs a ResNet-101 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
        
        return model
    
    def resnet152(self, **kwargs):
        """Constructs a ResNet-152 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
        
        return model
