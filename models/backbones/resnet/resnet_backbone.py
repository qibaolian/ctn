from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from models.backbones.resnet.resnet_models import ResNetModels

class NormalResnetBackbone(nn.Module):
    def __init__(self, orig_resnet):
        super(NormalResnetBackbone, self).__init__()
        
        self.channels = orig_resnet.channels
        self.first_stride = orig_resnet.first_stride
        # take pretrained resnet, except AvgPool and FC
        #self.head = orig_resnet.head
        self.layer0 = orig_resnet.conv
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def get_num_features(self):
        return self.channels[-1]

    def forward(self, x):
        tuple_features = list()
        #x = self.head(x)
        
        x = self.layer0(x)
        tuple_features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)

        return tuple_features

class DilatedResnetBackbone(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, multi_grid=(1, 2, 4)):
        super(DilatedResnetBackbone, self).__init__()
        
        self.channels = orig_resnet.channels
        self.first_stride = orig_resnet.first_stride
        
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            if multi_grid is None:
                orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet.layer4[i].apply(partial(self._nostride_dilate, dilate=int(4 * r)))

        elif dilate_scale == 16:
            if multi_grid is None:
                orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet.layer4[i].apply(partial(self._nostride_dilate, dilate=int(2 * r)))

        # Take pretrained resnet, except AvgPool and FC
        #self.head = orig_resnet.head
        self.layer0 = orig_resnet.conv
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            elif m.stride == (2, 2, 2):
                m.stride = (1, 1, 1)
                if m.kernel_size == (3, 3, 3):
                    m.dilation = (dilate // 2, dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2, dilate // 2)                
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
                elif m.kernel_size == (3, 3, 3):
                    m.dilation = (dilate, dilate, dilate)
                    m.padding = (dilate, dilate, dilate)

    def get_num_features(self):
        return self.channels[-1]

    def forward(self, x):
        tuple_features = list()
        #x = self.head(x)
        x = self.layer0(x)
        tuple_features.append(x)
        x = self.maxpool(x)        
        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)

        return tuple_features

class ResNetBackbone(object):
    def __init__(self, arch):
        self.resnet_models = ResNetModels()
        self.arch = arch
        
    def __call__(self, **kwargs):
        
        multi_grid = None
        #if self.configer.exists('network', 'multi_grid'):
        #    multi_grid = self.configer.get('network', 'multi_grid'
        
        if self.arch == 'resnet34':
            orig_resnet = self.resnet_models.resnet34(**kwargs)
            arch_net = NormalResnetBackbone(orig_resnet)

        elif self.arch == 'resnet34_dilated8':
            orig_resnet = self.resnet_models.resnet34(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        elif self.arch == 'resnet34_dilated16':
            orig_resnet = self.resnet_models.resnet34(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)

        elif self.arch == 'resnet50':
            orig_resnet = self.resnet_models.resnet50(**kwargs)
            arch_net = NormalResnetBackbone(orig_resnet)

        elif self.arch == 'resnet50_dilated8':
            orig_resnet = self.resnet_models.resnet50(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        elif self.arch == 'resnet50_dilated16':
            orig_resnet = self.resnet_models.resnet50(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)

        elif self.arch == 'resnet101':
            orig_resnet = self.resnet_models.resnet101(**kwargs)
            arch_net = NormalResnetBackbone(orig_resnet)

        elif arch == 'resnet101_dilated8':
            orig_resnet = self.resnet_models.resnet101(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        elif self.arch == 'resnet101_dilated16':
            orig_resnet = self.resnet_models.resnet101(**kwargs)
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)

        else:
            raise Exception('Architecture undefined!')

        return arch_net