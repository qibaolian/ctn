
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from loss.modules.loss_modules import *
from utils.config import cfg

LOSS_DICT = {
    'ce_loss': FCCELoss,
    'wce_loss': WCELoss,
    'bdice_loss': BinaryDiceLoss,
    'bce_loss': BCELoss,
    'focal_loss': FocalLoss,
    'dice_loss': DiceLoss,
    'gdice_loss': GateDiceLoss,
    'bbdice_loss': Bootstrapped_BinaryDiceLoss,
    'focaldice_loss': FocalDiceLoss,
    'eldice_loss': ELDiceLoss,
    'lovasz_loss': LovaszLoss,
    'hd_loss': HDLoss,
}

class LossManager(object):
    
    
    #def _parallel(self, loss):
    #    if self.configer.get('network', 'loss_balance') and len(range(torch.cuda.device_count())) > 1:
    #        from extensions.parallel.data_parallel import DataParallelCriterion
    #        loss = DataParallelCriterion(loss)
    # return loss
    
    def get_loss(self, loss_type):
        
        if '+' in loss_type:
            loss = nn.ModuleList([])
            loss_list = loss_type.split('+')
            for t in loss_list:
                if t not in LOSS_DICT:
                    raise NameError('Unkown loss type')
                loss.append(LOSS_DICT[t]())
            loss = MixLoss(loss)
        else:
            if loss_type not in LOSS_DICT:
                raise NameError('Unkown loss type')
        
            loss = LOSS_DICT[loss_type]()
        
        return loss
    
    def __call__(self):
        
        return self.get_loss(cfg.LOSS.TYPE)
            
        
        
        
        