from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np

from utils.config import cfg
from tasks.aneurysm.aneurysm_cls import AneurysmCls
from tasks.aneurysm.aneurysm_seg import AneurysmSeg
from tasks.aneurysm.blood_seg import BloodSeg
from tasks.ccta.ccta_vessel import CCTAVesselSeg
#from tasks.ccta_cl.ccta_cl import CCTACLSeg
from tasks.ccta_prior.ccta_prior import CCTAPriorSeg
#from tasks.ccta_global.ccta_global import CCTAGlobalSeg
from tasks.aneurysm.empty_task import EmptyTask
from tasks.intracranial_vessel.intra_vessel import INTRAVesselSeg
#from tasks.cta_vessel_split.model import CTAVesselSplit
from tasks.intra_vessel_prior.intra_vessel_prior import INTRAVesselPriorSeg

class Trainer(object):

    def __init__(self, task):

        self.task = task

    def train(self):

        f = open(os.path.join(cfg.OUTPUT_DIR,
                              '%04d_eval.txt' % self.task.start_epoch), 'w')
        first = True
        non_update = 0
        for epoch in range(self.task.start_epoch, cfg.SOLVER.EPOCHS):
            # self.task.adjust_learning_rate(epoch)
            #vv = self.task.validate()
            torch.cuda.empty_cache()
            #vv = self.task.validate()
            # train for one epoch
            self.task.train(epoch)
            # evaluate on validation dataset
            if epoch < cfg.TRAIN.START_VALIDATE or epoch % cfg.TRAIN.VALIDATE_FREQUENCE != 0:
                self.task.save_model(epoch, {})
                continue
            vv = self.task.validate()
            update = self.task.save_model(epoch, vv)
            non_update = 0 if update else non_update + 1
              
            if first:
                f.write('epoch')
                for k in vv.keys():
                    f.write('\t' + k)
                f.write('\n')
                first = False

            f.write('%04d' % epoch)
            for k in vv.keys():
                f.write('\t%.4f' % vv[k])
            f.write('\n')
            f.flush()

            #if non_update >= 20:
            #    break

def main_worker(args):
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    task = None

    if cfg.TASK.NAME == 'aneurysm_cls':
        task = AneurysmCls()
    elif cfg.TASK.NAME == 'aneurysm_seg':
        task = AneurysmSeg()
    elif cfg.TASK.NAME == 'blood_seg':
        task = BloodSeg()
    elif cfg.TASK.NAME == 'ccta_vessel':
        task = CCTAVesselSeg()             ###  
    elif cfg.TASK.NAME == 'ccta_cl':
        task = CCTACLSeg()
    elif cfg.TASK.NAME == 'ccta_prior':
        task = CCTAPriorSeg()
    elif cfg.TASK.NAME == 'ccta_global':
        task = CCTAGlobalSeg()
    elif cfg.TASK.NAME == 'intra_vessel':
        task = INTRAVesselSeg()
    elif cfg.TASK.NAME == 'cta_vessel_split':
        task = CTAVesselSplit()
    elif cfg.TASK.NAME == 'intra_vessel_prior':
        task = INTRAVesselPriorSeg()
    elif cfg.TASK.NAME == 'empty_task':
        task = EmptyTask()
    else:
        raise NameError('Unknown Task')
    cudnn.benchmark = True
    if args.train:
        trainer = Trainer(task)
        trainer.train()
    elif args.test:
        task.test()
