from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.backbones.resnet.resnet_backbone import ResNetBackbone
from models.backbones.resnext.resnext_backbone import ResNextBackbone
from models.backbones.densenet.densenet_backbone import DenseNetBackbone
from utils.config import cfg

class BackboneSelector(object):
    
    def __init__(self):
        self.arch = cfg.MODEL.BACKBONE.ARCH
    
    def __call__(self):
        
        model = None
        kwargs = {'input_channel': cfg.MODEL.INPUT_CHANNEL,
                  'head': cfg.MODEL.BACKBONE.HEAD }
        if 'resnet' in self.arch:
            model = ResNetBackbone(self.arch)(**kwargs)
        elif 'resnext' in self.arch:
            model = ResNextBackbone(self.arch)(**kwargs)
        elif 'densenet' in self.arch:
            model = DenseNetBackbone(self.arch)(**kwargs)
        else:
            raise NameError('unknown backbone')
        
        return model