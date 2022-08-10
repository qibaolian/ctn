
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.cls.nets.backbone_net import ResNet, DenseNet
from utils.config import cfg

CLS_MODEL_DICT = {
    'resnet': ResNet,
    'DenseNet': DenseNet,
}
    
class _ModelManager(object):
    
    def __call__(self):
        
        model_name = cfg.MODEL.NAME
        num_classes = cfg.MODEL.NCLASS
        
        if model_name not in CLS_MODEL_DICT:
            raise NameError('%s not define' % model_name)
        else:
            model = CLS_MODEL_DICT[model_name](num_classes)
        
        return model
        