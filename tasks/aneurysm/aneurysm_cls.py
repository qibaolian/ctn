import time
import numpy as np
import torch
from torch.autograd import Variable
from tasks.task import Task
from metrics.AUCEval import aucEval
from utils.tools.util import AverageMeter

from tasks.aneurysm.aneurysm_loader import AneurysmDataset, ValidatePatch3DLoader

from utils.config import cfg

def worker_init_fn(worker_id):
    np.random.seed(cfg.SEED)

class AneurysmCls(Task):
    
    def __init__(self):
        super(AneurysmCls, self).__init__()
        
        #def worker_init_fn(worker_id):
        #    np.random.seed(cfg.SEED)
        #kwargs = {'worker_init_fn': worker_init_fn, 'num_workers': cfg.TRAIN.DATA.WORKERS, 'pin_memory': True, 'drop_last':True}
        #train_set = Patch3DLoader(cfg.TRAIN.DATA.TRAIN_LIST)
        #self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.TRAIN.DATA.BATCH_SIZE, 
        #                                                shuffle=True, **kwargs)
            
        #val_set = Patch3DLoader(cfg.TRAIN.DATA.VAL_LIST, train=False)
        #self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.TRAIN.DATA.BATCH_SIZE,
        #                                              shuffle=False, **kwargs)
        
        #self.logger.info('Total train data: %d' % len(self.train_loader))
        #self.logger.info('Total validate data: %d' % len(self.val_loader))
        
        with open(cfg.TRAIN.DATA.TRAIN_LIST, 'r') as f:
            lines = f.readlines()
            subjects = [line.strip() for line in lines]
        self.data_set = AneurysmDataset(subjects)
        
        val_set = ValidatePatch3DLoader(cfg.TRAIN.DATA.VAL_LIST)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.TRAIN.DATA.BATCH_SIZE,
                                                      shuffle=False, num_workers=cfg.TRAIN.DATA.WORKERS,
                                                      pin_memory=True)
          
        self.logger.info('Total validate data: %d' % len(self.val_loader))
        
    def train(self, epoch):
        
        self.net.train()
        meter_names = ['loss', 'time']
        meters = {name: AverageMeter() for name in meter_names}
        aucEvalTrain = aucEval(cfg.MODEL.NCLASS)
        
        train_set = self.data_set.sample(200)
        kwargs = {'worker_init_fn': worker_init_fn, 'num_workers': cfg.TRAIN.DATA.WORKERS, 
                  'pin_memory': True, 'drop_last':True}
        
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.TRAIN.DATA.BATCH_SIZE, 
                                                   shuffle=True, **kwargs)
        self.logger.info('epoch=%04d, total train data: %d' % (epoch, len(self.train_loader)))
        
        t0 = time.time()
        for batch_idx, (image, label, _) in enumerate(self.train_loader):
            
            target = label.long()
            image, target = image.cuda(), target.cuda()
            image, target = Variable(image), Variable(target)
            
            out = self.net(image)
            loss = self.criterion(out, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            t1 = time.time()
            aucEvalTrain.addBatch(out.max(1)[1].data, target.data)
            meters['loss'].update(loss.data[0], image.size(0))
            meters['time'].update(t1-t0)
            
            if batch_idx % cfg.TRAIN.PRINT == 0:
                acc, pr, nr = aucEvalTrain.getMetric()
                self.logger.info('epoch=%04d, batch_idx=%06d, Time=%.2fs, loss=%.6f, Acc=%.4f%%, PR=%.4f%%, NR=%.4f%%' % \
                                 (epoch, batch_idx, meters['time'].avg, meters['loss'].avg, acc, pr, nr))
            
            t0 = time.time()
    
    def validate(self):
        
        self.net.eval()
        meter_names = ['time', 'loss']
        meters = {name: AverageMeter() for name in meter_names}
        aucEvalVal = aucEval(cfg.MODEL.NCLASS)
        
        t0 = time.time()
        for i, (image, label, _) in enumerate(self.val_loader):
            target = label.long()
            image, target = image.cuda(), target.cuda()
            image, target = image.cuda(), target.cuda()
            image, target = Variable(image, volatile=True), Variable(target, volatile=True)
            
            out = self.net(image)
            loss = self.criterion(out, target)
            aucEvalVal.addBatch(out.max(1)[1].data, target.data)
            t1 = time.time()
            
            meters['loss'].update(loss.data[0], image.size(0))
            meters['time'].update(t1-t0)
            
            t0 = time.time()
        
        acc, pr, nr = aucEvalVal.getMetric()
        self.logger.info('Validate: Time=%.3fms/batch, Loss=%.6f, Acc=%.4f%%, PR=%.4f%%, NR=%.4f%%' % \
                         (meters['time'].avg * 1000, meters['loss'].avg, acc, pr, nr))
        
        return {'Acc':acc, 'PR':pr, 'NR':nr}
    
    @staticmethod
    def add_args(parser):
           
        parser.add_argument('--loss', default='entropy', help='loss type: entropy, focalloss')
        parser.add_argument('--model', default='resnet34', help='model name')
        parser.add_argument('--pretrain', default=None, help='path where to load checkpoint')
        
        return parser