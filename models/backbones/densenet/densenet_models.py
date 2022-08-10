
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.tools.module_helper import CONV, BN, MAXPOOL, AVGPOOL


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', BN(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', CONV(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', BN(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', CONV(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
    
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
    
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', BN(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', CONV(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))


class DenseNetHeadA(nn.Module):
    def __init__(self, input_channel, num_init_features):
        super(DenseNetHeadA, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', CONV(input_channel, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', BN(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', MAXPOOL(kernel_size=3, stride=2, padding=1)),
        ]))
        
    def forward(self, x):
        return self.conv(x)
        
class DenseNetHeadB(nn.Module):
    def __init__(self, input_channel, num_init_features):
        super(DenseNetHeadB, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', CONV(input_channel, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
        ]))
    
    def forward(self, x):
        return self.conv(x)
    

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    
    def __init__(self, input_channel=3, head='A', growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()
        
        # First convolution
        if head == 'A':
            self.head = DenseNetHeadA(input_channel, num_init_features)
        elif head == 'B':
            self.head = DenseNetHeadB(input_channel, num_init_features)
        else:
            self.head = None
        
        self.features = nn.Sequential(OrderedDict([]))
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                avg_pool = AVGPOOL(kernel_size=2, stride=2)
                self.features.add_module('transition%d' % (i + 1), trans)
                self.features.add_module('transition%s_pool' % (i + 1), avg_pool)
                num_features = num_features // 2
        
        self.num_features = num_features
        # Final batch norm
        self.features.add_module('norm5', BN(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # Linear layer
        self.avgPool = AVGPOOL(7, stride=1)
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                #nn.init.kaiming_normal_(m.weight)
                n = reduce((lambda x, y: x * y), m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                #nn.init.constant_(m.weight, 1)
                #nn.init.constant_(m.bias, 0)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                #nn.init.constant_(m.bias, 0)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.head(x)
        features = self.features(x)
        out = self.avgpool(out).view(features.size(0), -1)                       
        out = self.classifier(out)
        return out
        
class DenseNetModels(object):

    def densenet121(self, **kwargs):
        r"""Densenet-121 model from
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = DenseNet(num_init_features=64, growth_rate=32,
                         block_config=(6, 12, 24, 16), **kwargs)
                                 
        return model

    def densenet169(self, **kwargs):
        r"""Densenet-169 model from
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = DenseNet(num_init_features=64, growth_rate=32,
                         block_config=(6, 12, 32, 32), **kwargs)

        return model

    def densenet201(self, **kwargs):
        r"""Densenet-201 model from
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = DenseNet(num_init_features=64, growth_rate=32,
                         block_config=(6, 12, 48, 32), **kwargs)

        return model

    def densenet161(self, **kwargs):
        r"""Densenet-161 model from
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = DenseNet(num_init_features=96, growth_rate=48,
                         block_config=(6, 12, 36, 24), **kwargs)

        return model    
        
        
    