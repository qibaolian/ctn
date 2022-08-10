# -*- coding=utf-8 -*-
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tasks.task import Task
from metrics.AUCEval import aucEval
from metrics.DiceEval import diceEval
from utils.tools.util import AverageMeter
from utils.tools.tensor import tensor_dice
from metrics.eval import eval_results

from skimage import measure
from .aneurysm_loader import AneurysmDataset, ValidatePatch3DLoaderMemory

from .nets.aneurysm_net import DAResUNet, DAResUNet2, DAResNet34, CENet
from utils.config import cfg
from utils.tools.util import count_time
from tasks.aneurysm.datasets.aneurysm_dataset import ANEURYSM_SEG

import threading


def worker_init_fn(worker_id):
    np.random.seed(cfg.SEED)


class EmptyTask(Task):

    def __init__(self):
        super(EmptyTask, self).__init__()

        if cfg.TASK.STATUS == 'train':
            self.data_sampler = AneurysmDataset(cfg.TRAIN.DATA.TRAIN_LIST)
            self.data_sampler.asyn_sample(50, 100)
            # self.data_sampler.asyn_sample(1, 2)  # for fast train
            self.val_loader = ValidatePatch3DLoaderMemory(cfg.TRAIN.DATA.VAL_LIST)

    def get_model(self):

        if cfg.MODEL.NAME == 'da_resunet':
            self.net = DAResUNet(cfg.MODEL.NCLASS, k=32)
        elif cfg.MODEL.NAME == 'da_resunet2':
            self.net = DAResUNet2(cfg.MODEL.NCLASS, k=24)
        elif cfg.MODEL.NAME == 'da_resnet34':
            self.net = DAResNet34(cfg.MODEL.NCLASS, k=32)
        elif cfg.MODEL.NAME == 'cenet':
            self.net = CENet(cfg.MODEL.NCLASS, k=16)
        else:
            super(EmptyTask, self).get_model()

    def train(self, epoch):
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_lr()[0]
        print("epoch:{} lr:{:.6f}".format(epoch,lr))
        # print("in train cur epoch:{}".format(epoch))

    def validate(self):
        #print("in val")
        return {
            'Loss': 1.0,
            'Dice': 1.0,
            'F1-Score': 1.0,
            'Recall': 1.0,
            'Precision': 1.0,
        }
