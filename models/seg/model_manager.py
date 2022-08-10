from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.seg.nets.deeplabv3_plus import DeepLabV3Plus
from models.seg.nets.pspnet import PSPNet
from models.seg.nets.unet import UNET
from models.seg.nets.backbone_unet import BackboneUNet
from tasks.aneurysm.nets.resunet import ResUNet, OCResUNet, DAResUNet, FCN, DAResNet3d, CENet, MIXNet
from tasks.aneurysm.nets.resunet2d import DAResNet2D
from utils.config import cfg

model_dict = {
    'pspnet': {'model': PSPNet, 'default_args': {'num_classes': 2}},
    'deeplabv3+': {'model': DeepLabV3Plus, 'default_args': {'num_classes': 2}},
    'backbone_unet': {'model': BackboneUNet, 'default_args': {'num_classes': 2}},
    'unet': {'model': UNET, 'default_args': {'input_channel': 3, 'num_classes': 2}},
    'resunet': {'model': ResUNet, 'default_args': {'num_classes': 2, 'k': 32, 'psp': False}},
    'resunet_psp': {'model': ResUNet, 'default_args': {'num_classes': 2, 'k': 32, 'psp': True}},
    'oc_resunet': {'model': OCResUNet, 'default_args': {'num_classes': 2, 'k': 32, 'psp': False}},
    'oc_resunet_psp': {'model': OCResUNet, 'default_args': {'num_classes': 2, 'k': 32, 'psp': True}},
    'da_resunet': {'model': DAResUNet, 'default_args': {'num_classes': 2, 'k': 32}},
    'fcn': {'model': FCN, 'default_args': {'num_classes': 2, 'k': 8}},
    'da_resnet34_2d': {'model': DAResNet2D, 'default_args': {'num_classes': 2, 'k': 32}},
    'da_resnet34_3d': {'model': DAResNet3d,
                       'default_args': {'input_channels': cfg.MODEL.INPUT_CHANNEL, 'num_classes': 2, 'k': 32, "deep_supervision": False, 'heat_map': False}},
    'cenet_3d': {'model': CENet,
                 'default_args': {'num_classes': 2, 'k': 32, "deep_supervision": False, 'heat_map': False}},
    'mixnet': {'model': MIXNet,
               'default_args': {'num_classes': 2, 'k': 32, "deep_supervision": False, 'heat_map': False}},
}


class _ModelManager(object):
    def __call__(self):
        model_name = cfg.MODEL.NAME
        if model_name not in model_dict.keys():
            raise NameError("Unknown Model Name:{}!".format(model_name))
        get_model = model_dict[model_name.lower()]['model']
        default_args = model_dict[model_name.lower()]['default_args']
        if cfg.MODEL.PARA:
            personal_args = {k.lower(): cfg.MODEL.PARA[k] for k in cfg.MODEL.PARA.keys()}
            default_args.update(personal_args)
        return get_model(**default_args)
