from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from models.backbones.resnext.resnext_models import ResNeXtModels

class NormalResnextBackbone(nn.Module):
    def __init__(self, orig_resnext):
        super(NormalResneXtBackbone, self).__init__()

        self.num_features = 1024
        # take pretrained resnext, except AvgPool and FC
        self.head = orig_resnext.head
        self.layer1 = orig_resnext.layer1
        self.layer2 = orig_resnext.layer2
        self.layer3 = orig_resnext.layer3
        self.layer4 = orig_resnext.layer4

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.head(x)

        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)

        return tuple_features

class DilatedResnextBackbone(nn.Module):
    def __init__(self, orig_resnext, dilate_scale=8, multi_grid=(1, 2, 4)):
        super(DilatedResnextBackbone, self).__init__()

        self.num_features = 2048
        from functools import partial

        if dilate_scale == 8:
            orig_resnext.layer3.apply(partial(self._nostride_dilate, dilate=2))
            if multi_grid is None:
                orig_resnext.layer4.apply(partial(self._nostride_dilate, dilate=4))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnext.layer4[i].apply(partial(self._nostride_dilate, dilate=int(4 * r)))

        elif dilate_scale == 16:
            if multi_grid is None:
                orig_resnext.layer4.apply(partial(self._nostride_dilate, dilate=2))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnext.layer4[i].apply(partial(self._nostride_dilate, dilate=int(2 * r)))

        # Take pretrained resnet, except AvgPool and FC
        self.head = orig_resnext.head
        self.layer1 = orig_resnext.layer1
        self.layer2 = orig_resnext.layer2
        self.layer3 = orig_resnext.layer3
        self.layer4 = orig_resnext.layer4

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
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.head(x)

        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)

        return tuple_features

class ResNextBackbone(object):
    def __init__(self, arch):
        self.resnext_models = ResNeXtModels()
        self.arch = arch
        
    def __call__(self, **kwargs):
        
        multi_grid = None
        #if self.configer.exists('network', 'multi_grid'):
        #    multi_grid = self.configer.get('network', 'multi_grid'
        
        if self.arch == 'resnext50':
            orig_resnext = self.resnext_models.resnext50(**kwargs)
            arch_net = NormalResnextBackbone(orig_resnext)

        elif self.arch == 'resnet50_dilated8':
            orig_resnext = self.resnext_models.resnext50(**kwargs)
            arch_net = DilatedResnextBackbone(orig_resnext, dilate_scale=8, multi_grid=multi_grid)

        elif self.arch == 'resnet50_dilated16':
            orig_resnext = self.resnext_models.resnet50(**kwargs)
            arch_net = DilatedResnextBackbone(orig_resnext, dilate_scale=16, multi_grid=multi_grid)

        elif self.arch == 'resnet101':
            orig_resnext = self.resnext_models.resnext101(**kwargs)
            arch_net = NormalResnextBackbone(orig_resnext)

        elif self.arch == 'resnet101_dilated8':
            orig_resnext = self.resnext_models.resnext101(**kwargs)
            arch_net = DilatedResnextBackbone(orig_resnext, dilate_scale=8, multi_grid=multi_grid)

        elif self.arch == 'resnet101_dilated16':
            orig_resnext = self.resnext_models.resnext50(**kwargs)
            arch_net = DilatedResnextBackbone(orig_resnext, dilate_scale=16, multi_grid=multi_grid)

        elif self.arch == 'resnet152':
            orig_resnext = self.resnext_models.resnext152(**kwargs)
            arch_net = NormalResnextBackbone(orig_resnext)

        elif arch == 'resnet152_dilated8':
            orig_resnext = self.resnext_models.resnext101(**kwargs)
            arch_net = DilatedResnextBackbone(orig_resnext, dilate_scale=8, multi_grid=multi_grid)

        elif self.arch == 'resnet152_dilated16':
            orig_resnext = self.resnext_models.resnext101(**kwargs)
            arch_net = DilatedResnextBackbone(orig_resnext, dilate_scale=16, multi_grid=multi_grid)

        else:
            raise Exception('Architecture undefined!')

        return arch_net