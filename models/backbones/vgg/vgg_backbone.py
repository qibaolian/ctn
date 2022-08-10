

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.backbones.vgg.vgg_models import VGGModels

class VGGBackbone(object):
    
    def __init__(self, arch):
        self.arch = arch
        self.vgg_models = VGGModels()
        
    def __call__(sefl, **kwargs):
        
        arch_net = self.vgg_models.vgg(self.arch, **kwargs)
        
        return arch_net