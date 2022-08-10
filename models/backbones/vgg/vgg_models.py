
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from models.tools.module_helper import CONV, BN, MAXPOOL, AVGPOOL

CONFIG_DICT = {
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg13_dilated8': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512],
    'vgg16_dilated8': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512],
    'vgg19_dilated8': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512],
    'vgg13_dilated16': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'vgg16_dilated16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512],
    'vgg19_dilated16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512],
}

def make_layer(cfg, in_channel):
    
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [MAXPOOL(kernel_size=2, stride=2)]
        
        elif v == 'C':
            layers += [MAXPOOL(kernel_size=2, stride=2, ceil_mode=True)]
        
        else:
            conv = CONV(in_channel, v, kernel_size=3, padding=1)
            layers += [conv, BN(v), nn.ReLU(inplace=True)]
            
            in_channel = v
    
    return nn.Sequential(*layers)

class VGG(nn.Module):
    
    def __init__(self, cfg_name, input_channel=3, num_classes=1000):
        super(VGG, self).__init__()
        self.num_features = 512
        vgg_cfg = CONFIG_DICT[cfg_name]
        self.features = make_layers(vgg_cfg, input_channel)
        
        self.avgPool = AVGPOOL(7, stride=1)
        self.fc = nn.Linear(self.num_features, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                n = reduce((lambda x, y: x * y), m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
     
    def forward(self, x):
        x = self.features(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class VGGModles(object):
    
    def vgg(self, cfg_name, **kwargs):
        
        model = VGG(cfg_name, **kwargs)
        
        return model
        
    
    
        
