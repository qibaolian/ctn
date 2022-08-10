import os
import time
import numpy as np
from skimage import measure
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tasks.task import Task
from metrics.DiceEval import diceEval
from utils.tools.util import AverageMeter
from utils.tools.tensor import tensor_dice

from .dataset import VesselDS, VesselDatasetSampler, VesselSubjectDS
from utils.config import cfg

from tasks.aneurysm.nets.vessel_net import SimpleNet, DAResNet18, DASEResNet18
from tasks.ccta.nets.coronary_net import SimpleCoronaryNet, SimpleCoronaryNet2, CoronaryUNet
from tasks.ccta.nets.unet_pse_cf import SEUNet, CFUNet, SECFUNet, SECFUNet3
from tasks.aneurysm.nets.resunet import DAResNet3d
from tasks.aneurysm.nets.MultiResUNet3D import UNet4, MultiResUnet4, MultiResUnet3, DAMultiResUnet3_4
from tasks.ccta.nets.unet_3d import SEUNet4, SECFUNet4, SCSECFUNet4, DASECFUNet4
from tasks.ccta.nets.vt_net import VT_Net, DASEResNet18_HP
from tasks.ccta.nets.mww_net import MWWNet

from utils.tools.util import count_time, progress_monitor
from metrics.eval_vessel import eval_volume, eval_xinji, eval_ccta_volume, eval_ccta_volume_v2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from loss.modules.loss_modules import *

from tasks.aneurysm.datasets.base_dataset import RandomSequenceSampler, DataLoaderX, data_prefetcher

def worker_init_fn(worker_id):
    np.random.seed(cfg.SEED)

class CCTAVesselSeg(Task):

    def __init__(self):
        super(CCTAVesselSeg, self).__init__()
        
        '''
        if cfg.TASK.STATUS == 'train':
            self.online_sample = 'subjects' in cfg.TRAIN.DATA.TRAIN_LIST
            if self.online_sample:
                self.data_sampler = VesselDatasetSampler(cfg.TRAIN.DATA.TRAIN_LIST)
                self.data_sampler.asyn_sample(20, 100)
            else:
                self.train_set = VesselDS({"train_list":cfg.TRAIN.DATA.TRAIN_LIST}, stage="train")
            self.val_subjects = 'subjects' in cfg.TRAIN.DATA.VAL_LIST
            if not self.val_subjects:
                para_dict = {"val_list": cfg.TRAIN.DATA.VAL_LIST}
                self.val_set = VesselDS(para_dict, stage="val")
        '''
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.bdice = BinaryDiceLoss().to(self.device)
        self.sdm = SDMLoss().to(self.device)
        
        
    def get_model(self):
        if cfg.MODEL.NAME == 'simple_net':
            self.net = SimpleNet(cfg.MODEL.NCLASS, k=16)
        elif cfg.MODEL.NAME == 'da_resnet3d':
            self.net = DAResNet3d(num_classes=cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'da_resnet18':
            self.net = DAResNet18(cfg.MODEL.NCLASS, k=16)
        elif cfg.MODEL.NAME == 'da_seresnet18':   ### da_seresnet18
            self.net = DASEResNet18(cfg.MODEL.PARA.NUM_CLASSES, k=24, input_channels=1,
                                    heatmap= cfg.TRAIN.DATA.USE_HEATMAP if 'USE_HEATMAP' in cfg.TRAIN.DATA.keys() else False)
        elif cfg.MODEL.NAME == 'da_seresnet18_hp':
            self.net = DASEResNet18_HP(cfg.MODEL.PARA.NUM_CLASSES, k=24, input_channels=1)
        elif cfg.MODEL.NAME == 'UNet4':
            self.net = UNet4(cfg.MODEL.PARA.NUM_CLASSES)
        elif cfg.MODEL.NAME == 'MultiResUnet4':
            self.net = MultiResUnet3(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'DAMultiResUnet3_4':
            self.net = DAMultiResUnet3_4(cfg.MODEL.PARA.NUM_CLASSES, attention=False)
        elif cfg.MODEL.NAME == 'coronary_simplenet':
            self.net = SimpleCoronaryNet(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'coronary_simplenet2':
            self.net = SimpleCoronaryNet2(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K, se=False)
        elif cfg.MODEL.NAME == 'coronary_unet':
            self.net = CoronaryUNet(cfg.MODEL.NCLASS, k=cfg.MODEL.K, psp=True)
        elif cfg.MODEL.NAME == 'se_unet':
            self.net = SEUNet(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'cf_unet':
            self.net = CFUNet(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'cfse_unet':
            self.net = SECFUNet(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'cfse_unet3':
            self.net = SECFUNet3(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'coronary_unet':
            self.net = CoronaryUNet(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K, psp=True)
        elif cfg.MODEL.NAME == 'se_unet4':
            self.net = SEUNet4(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'secf_unet4':
            self.net = SECFUNet4(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'scsecf_unet4':
            self.net = SCSECFUNet4(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'dasecf_unet4':
            self.net = DASECFUNet4(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'vt_net':
            self.net = VT_Net(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K, add=False)
        elif cfg.MODEL.NAME == 'mww_net':
            self.net = MWWNet(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K,
                             heatmap= cfg.TRAIN.DATA.USE_HEATMAP if 'USE_HEATMAP' in cfg.TRAIN.DATA.keys() else False)
        else:
            super().get_model()  ### 'resnet'
    
    @count_time
    def train(self, epoch):
        self.train_patches(epoch)
        #self.train_sdf(epoch)
        #self.train_npz_2(epoch)
        
    def train_npz(self, epoch):
        self.net.train()
        meter_names = ['time', 'loss', 'seg', 'boundary', 'sdm', 'bdice']
        meters = {name: AverageMeter() for name in meter_names}
        diceEvalTrain = diceEval(cfg.MODEL.PARA.NUM_CLASSES)
        diceSDMEvalTrain = diceEval(cfg.MODEL.NCLASS)
        
        if self.online_sample:
            self.train_set = self.data_sampler.get_data_loader()
            self.data_sampler.asyn_sample(200, 10)
        
        kwargs = {'worker_init_fn': worker_init_fn, 'num_workers': cfg.TRAIN.DATA.WORKERS,
                  'pin_memory': True, 'drop_last': True}
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=cfg.TRAIN.DATA.BATCH_SIZE,
                                                        shuffle=True, **kwargs)
        self.logger.info('epoch=%04d, total train data: %d' % (epoch, len(self.train_loader)))
        t0 = time.time()
        self.logger.info("current epoch learning rate:{:.8f}!".format(self.lr_scheduler.get_lr()[0]))
        
        for batch_idx, data in enumerate(self.train_loader):

            image, mask = data['img'].to(self.device), data['gt'].long().to(self.device)
            self.optimizer.zero_grad()
            out = self.net(image)            
            seg = out['y']
            loss = 0
            seg_loss = self.criterion(seg, mask)    
            loss +=  seg_loss
            meters['seg'].update(seg_loss.item(), image.size(0))
            
            if 'sdm' in data:
                sdm = data['sdm'].to(self.device)
                bd_loss = 10.0 * ( sdm * seg).mean()
                loss += bd_loss             
                meters['boundary'].update(bd_loss.item(), image.size(0))
                
                if 'sdm' in out:
                    sdm_loss = 2.0 * self.sdm(out['sdm'], sdm)
                    loss += sdm_loss
                    meters['sdm'].update(sdm_loss.item(), image.size(0))
                    
                    output_soft = torch.sigmoid(-1500*out['sdm'])
                    bdice_loss = self.bdice(output_soft, mask)
                    loss += bdice_loss
                    meters['bdice'].update(bdice_loss.item(), image.size(0))
            
            loss.backward()
            self.optimizer.step()
            
            if batch_idx % 10 == 0:
                #diceEvalTrain.addBatch((seg>=0.5).long(), mask)
                diceEvalTrain.addBatch(seg.max(1)[1], mask)
                if 'sdm' in data:
                    diceSDMEvalTrain.addBatch((output_soft>=0.5).long(), mask)

            t1 = time.time()
            meters['time'].update(t1-t0)
            meters['loss'].update(loss.item(), image.size(0))

            if batch_idx % cfg.TRAIN.PRINT == 0:
                d0, d1 = diceEvalTrain.getMetric(), diceSDMEvalTrain.getMetric()
                self.logger.info('epoch=%04d, batch_idx=%06d, Time=%.2fs, loss=(total:%.6f, seg:%.6f, boundary:%.6f, sdm:%.6f, bdice:%.6f),  dice=(seg: %.4f, sdm: %.4f)' % \
                             (epoch, batch_idx, meters['time'].avg, meters['loss'].avg,
                              meters['seg'].avg, meters['boundary'].avg, meters['sdm'].avg, meters['bdice'].avg, d0, d1))

            t0 = time.time()
         
        if cfg.SOLVER.LR_MODE == 'plateau':
            self.lr_scheduler.step(meters['loss'].avg)
        else:
            self.lr_scheduler.step()
    
    def train_sdf(self, epoch):
        
        #mse_loss = nn.MSELoss(reduction='mean').to(self.device)
        mse_loss = nn.SmoothL1Loss(reduction='mean').to(self.device)
        
        with open(cfg.TRAIN.DATA.TRAIN_LIST, 'r') as f:
            subjects = f.readlines()
            subjects = [subject.strip() for subject in subjects]
        
        batch_size = cfg.TRAIN.DATA.BATCH_SIZE
        sample_per_subject = batch_size * 1
        train_set = VesselSubjectDS({"subjects":subjects, "sample": sample_per_subject}, stage="train")
        
        self.net.train()
        meter_names = ['time', 'loss']
        meters = {name: AverageMeter() for name in meter_names}
        
        kwargs = {'batch_size': batch_size, 'shuffle': False,
                  'num_workers': cfg.TRAIN.DATA.WORKERS, #'worker_init_fn': worker_init_fn, 
                  'pin_memory': True, 'drop_last': True,
                  'sampler': RandomSequenceSampler(len(subjects), sample_per_subject, batch_size, cfg.TRAIN.DATA.WORKERS)}
        train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
        self.logger.info('epoch=%04d, total train data: %d' % (epoch, len(train_loader)))
        t0 = time.time()
        self.lr_scheduler.step()
        self.logger.info("current epoch learning rate:{:.8f}!".format(self.lr_scheduler.get_lr()[0]))
        
        
        for batch_idx, data in enumerate(train_loader):

            image, mask = data['img'].to(self.device), data['gt'].to(self.device)
            
            out = self.net(image)            
            out = torch.tanh(out['y'])
            loss = mse_loss(out, mask)
            meters['loss'].update(loss.item(), image.size(0))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            t1 = time.time()
            meters['time'].update(t1-t0)
            meters['loss'].update(loss.item(), image.size(0))

            if batch_idx % cfg.TRAIN.PRINT == 0:
                self.logger.info('epoch=%04d, batch_idx=%06d, Time=%.2fs, loss=%.6f' % \
                             (epoch, batch_idx, meters['time'].avg, meters['loss'].avg))

            t0 = time.time()
        
    def train_npz_2(self, epoch):
        
        self.net.train()
        meter_names = ['time', 'loss']
        meters = {name: AverageMeter() for name in meter_names}
        diceEvalTrain = diceEval(cfg.MODEL.PARA.NUM_CLASSES)
        
        train_set = VesselDS({"train_list":cfg.TRAIN.DATA.TRAIN_LIST}, stage="train")
        kwargs = {'num_workers': cfg.TRAIN.DATA.WORKERS,
                       'pin_memory': True, 'drop_last': True}
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.TRAIN.DATA.BATCH_SIZE, shuffle=True, **kwargs)
        self.logger.info('epoch=%04d, total train data: %d' % (epoch, len(train_loader)))
        t0 = time.time()
        self.lr_scheduler.step()
        self.logger.info("current epoch learning rate:{:.8f}!".format(self.lr_scheduler.get_lr()[0]))
        
        for batch_idx, data in enumerate(train_loader):

            image, mask = data['img'].to(self.device), data['gt'].long().to(self.device)
            out = self.net(image)            
            seg = out['y']
            loss = 0
            seg_loss = self.criterion(seg, mask)    
            loss +=  seg_loss
            meters['loss'].update(seg_loss.item(), image.size(0))
            
            if 'rd' in data:
                rd = data['rd'].to(self.device)
                pp = seg[:, 1, ...]
                rd, pp = rd.contiguous().view(rd.size(0), -1), pp.contiguous().view(pp.size(0), -1)
                num = torch.sum(rd*pp, dim=1)
                den1 = torch.sum(pp*pp, dim=1)
                den2 = torch.sum(rd*rd, dim=1)
                dd = (2.0 * num + 1e-4) / ( den1 + den2 + 1e-4)
                dd =1.0 -  torch.sum(dd) / dd.size(0)
                loss += dd
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if batch_idx % 10 == 0:
                diceEvalTrain.addBatch(seg.max(1)[1], mask)

            t1 = time.time()
            meters['time'].update(t1-t0)
            #meters['loss'].update(loss.item(), image.size(0))

            if batch_idx % cfg.TRAIN.PRINT == 0:
                d = diceEvalTrain.getMetric()
                self.logger.info('epoch=%04d, batch_idx=%06d, Time=%.2fs, loss=%.6f,  dice=%.4f' % \
                             (epoch, batch_idx, meters['time'].avg, meters['loss'].avg, d))

            t0 = time.time()

    def train_patches(self, epoch):
        
        if 'prob' in cfg.TRAIN.DATA.TRAIN_LIST:
            with open(cfg.TRAIN.DATA.TRAIN_LIST, 'r') as f:
                lines = f.readlines()
                lines = [line.strip().split(' ') for line in lines]
                subjects = [line[0] for line in lines]
                probs = np.array([float(line[1]) for line in lines])
                probs = (probs / probs.sum()).tolist()
        else:
            with open(cfg.TRAIN.DATA.TRAIN_LIST, 'r') as f:
                subjects = f.readlines()
                subjects = [subject.strip() for subject in subjects]
                probs = None
        
        sample_num_subjects =  cfg.TRAIN.DATA.SAMPLE_NUM_SUBJECTS if 'SAMPLE_NUM_SUBJECTS' in cfg.TRAIN.DATA.keys() else -1
        
        if sample_num_subjects > 0:
            k = min(sample_num_subjects, len(subjects))
            subjects = np.random.choice(subjects, size=k, p=probs, replace=probs!=None)
        
        batch_size = cfg.TRAIN.DATA.BATCH_SIZE
        sample_per_subject = batch_size * (cfg.TRAIN.DATA.SAMPLE_PER_SUBJECT if 'SAMPLE_PER_SUBJECT' in cfg.TRAIN.DATA.keys() else 1)
        sampler = RandomSequenceSampler(len(subjects), sample_per_subject, batch_size, cfg.TRAIN.DATA.WORKERS, 
                                                             max_subject_in_batch = 8)
        train_set = VesselSubjectDS({"subjects":subjects, "sample": sample_per_subject,
                                                 'subjects_per_worker': sampler.subjects_per_worker}, 
                                                 stage="train")
        
        self.net.train()
        meter_names = ['time', 'loss', 'skel_loss', 'hp_loss', 'hp_dice']
        meters = {name: AverageMeter() for name in meter_names}
        diceEvalTrain = diceEval(cfg.MODEL.PARA.NUM_CLASSES)
        
        kwargs = {'batch_size': batch_size, 'shuffle': False,
                       'num_workers': cfg.TRAIN.DATA.WORKERS, #'worker_init_fn': worker_init_fn, 
                       'pin_memory': True, 'drop_last': True,
                       'sampler': sampler}
        #train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
        train_loader = DataLoaderX(train_set, **kwargs)
        self.logger.info('epoch=%04d, total train data: %d' % (epoch, len(train_loader)))
        t0 = time.time()
        #self.lr_scheduler.step()
        #if cfg.SOLVER.LR_MODE != 'plateau':
        self.logger.info("current epoch learning rate:{:.8f}!".format(self.lr_scheduler.get_lr()[0]))
        
        heatmap = cfg.TRAIN.DATA.USE_HEATMAP if 'USE_HEATMAP' in cfg.TRAIN.DATA.keys() else False
        if heatmap:
            mse_loss = nn.SmoothL1Loss(reduction='mean').to(self.device)
            bdice = BinaryDiceLoss().to(self.device)
        
        skeleton = cfg.TRAIN.DATA.USE_SKELETON if 'USE_SKELETON' in cfg.TRAIN.DATA.keys() else False
        if skeleton:
            ce_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(cfg.LOSS.CLASS_WEIGHT).cuda(), ignore_index=255)
        
        prefetch_train_loader = data_prefetcher(train_loader)
        for batch_idx, data in enumerate(prefetch_train_loader):

            #image, mask = data['img'].to(self.device), data['gt'].long().to(self.device)
            image, mask = data['img'], data['gt'].long()
            out = self.net(image)            
            seg = out['y']
            loss = 0
            if isinstance(seg, tuple):
                ww = [1.0, 0.5, 0.25, 0.15, 0.1]
                seg_loss = 0
                for i, pred in enumerate(seg):
                    if mask.size()[-3:] == pred.size()[-3:]:
                        seg_loss += ww[i] * self.criterion(pred, mask)
                    else:
                        sw, sh, sd = mask.size()[-3:]
                        dw, dh, dd = pred.size()[-3:]
                    
                        gt = F.max_pool3d(mask.float(), (sw//dw, sh//dh, sd//dd))
                        seg_loss += ww[i] * self.criterion(pred, gt.long())
                seg = seg[0]                    
            else:
                seg_loss = self.criterion(seg, mask)
            loss +=  seg_loss
            
            if skeleton:
                #skel = data['skel'].long().to(self.device)
                skel = data['skel'].long()
                skel_loss = ce_loss(seg, skel)
                loss += 0.5*skel_loss
                meters['skel_loss'].update(0.5*skel_loss.item(), image.size(0))
            
            if heatmap:
                #hp = data['hp'].to(self.device)
                hp = data['hp']
                hp_loss = mse_loss(out['hp'], hp)
                loss += 20.0*hp_loss
                meters['hp_loss'].update(20.0*hp_loss.item(), image.size(0))
                
                hp_soft = torch.sigmoid(1000*(out['hp']-0.99))
                bdice_loss = self.bdice(hp_soft[:, 0, ...], mask>0)
                loss += 0.1*bdice_loss
                meters['hp_dice'].update(0.1*bdice_loss.item(), image.size(0))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            meters['loss'].update(loss.item(), image.size(0))
            if batch_idx % 3 == 0:
                if seg.size()[-3:] != mask.size()[-3:]:
                    seg = F.interpolate(seg, mask.size()[-3:], mode='trilinear', align_corners=True)
                diceEvalTrain.addBatch(seg.max(1)[1], mask)

            t1 = time.time()
            meters['time'].update(t1-t0)

            if batch_idx % cfg.TRAIN.PRINT == 0:
                dd = diceEvalTrain.getMetric()
                ss = 'class: (' + ','.join('%.4f' % d for d in dd[:-1]) + ')'
                ss += ' mean: %.4f, total: %.4f' % (dd[:-1].mean(), dd[-1])
                
                ss2 = ''
                seg = meters['loss'].avg
                if skeleton:
                    ss2 += 'skel:%.6f, ' % meters['skel_loss'].avg
                    seg -= meters['skel_loss'].avg
                if heatmap:
                    ss2 += 'hp:%.6f, ' % meters['hp_loss'].avg
                    seg -= meters['hp_loss'].avg
                    ss2 += 'hp_dice:%.4f, ' % meters['hp_dice'].avg
                    seg -=  meters['hp_dice'].avg
                ss2 += 'seg:%.6f, total:%.6f' % (seg, meters['loss'].avg)
                
                self.logger.info('epoch=%04d, batch_idx=%06d, Time=%.2fs, loss=[%s],  dice=[%s]' % \
                             (epoch, batch_idx, meters['time'].avg, ss2, ss))
            t0 = time.time()
        
        if cfg.SOLVER.LR_MODE == 'plateau':
            self.lr_scheduler.step(meters['loss'].avg)
        else:
            self.lr_scheduler.step()

    @count_time
    def validate(self):
        #if self.val_subjects:
        #    return self.validate_subjects()
        #else:
        #    return self.validate_patches()
        return self.validate_skeleton_subjects()
        return self.validate_sdf()
    
    def validate_sdf(self, stage='val'):
        
        #mse_loss = nn.MSELoss(reduction='mean').to(self.device)
        mse_loss = nn.SmoothL1Loss(reduction='mean').to(self.device)
        
        val_path = cfg.TRAIN.DATA.VAL_LIST if stage == 'val' else cfg.TEST.DATA.TEST_FILE
        with open(val_path, 'r') as f:
            subjects = f.readlines()
            subjects = [subject.strip() for subject in subjects]
            
        self.logger.info("the number of subjects to be inferenced is %d" % len(subjects))        
        
        self.net.eval()
        meter_names = ['loss', 'time']
        meters = {name: AverageMeter() for name in meter_names}
        
        seg_results = []
        for subject in subjects:
            self.logger.info("start to process %s data" % subject)
            para_dict = {"subject": subject}
            test_set = VesselSubjectDS(para_dict, stage="val")
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
                                                      shuffle=False, pin_memory=True)
            with torch.no_grad():
                v_x, v_y, v_z = test_set.volume_size()[-3:]
                seg = torch.FloatTensor(cfg.MODEL.PARA.NUM_CLASSES, v_x, v_y, v_z).zero_().to(self.device)
                num = torch.ByteTensor(v_x, v_y, v_z).zero_().to(self.device)
                
                for _, data in enumerate(test_loader):
                    img = data['img'].to(self.device)
                    msk = data['gt'].to(self.device)
                    coord = data['coord']
                    out = self.net(img)
                    pred = torch.tanh(out['y'])
                    loss = mse_loss(pred, msk)
                    meters['loss'].update(loss.item(), img.size(0))
                    
                    for idx in range(img.size(0)):
                        sx, ex = coord[idx][0][0], coord[idx][0][1]
                        sy, ey = coord[idx][1][0], coord[idx][1][1]
                        sz, ez = coord[idx][2][0], coord[idx][2][1]
                        seg[:, sx:ex, sy:ey, sz:ez] += pred[idx]
                        #seg[sx:ex, sy:ey, sz:ez] = torch.max(seg[sx:ex, sy:ey, sz:ez], pred[idx][1])
                        num[sx:ex, sy:ey, sz:ez] += 1
                seg = (seg / num.float()).cpu().numpy()
                #seg = seg.cpu().numpy()
                seg_results.append((pow(seg-test_set.volume_mask(), 2).mean()))

                #save segment results
                if cfg.TEST.SAVE:
                    self.logger.info('save %s prediction results' % subject)
                    test_set.save(seg, os.path.join(cfg.TEST.SAVE_DIR, '%s_seg.nii.gz' % subject), True) 
        
        self.logger.info('Validate: loss=%.6f, diff=%.4f' % (meters['loss'].avg, np.mean(seg_results)))
        
        return {'Diff': 1.0-np.mean(seg_results), 'Loss': -meters['loss'].avg}        
    
    def validate_patches(self):
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
                                                                      shuffle=False, pin_memory=True)
        self.net.eval()
        meter_names = ['loss', 'time', 'mse']
        meters = {name: AverageMeter() for name in meter_names}
        diceEvalVal = diceEval(cfg.MODEL.PARA.NUM_CLASSES)

        with torch.no_grad():
            t0 = time.time()
            for i, data in enumerate(self.val_loader):

                image, mask = data['img'].to(self.device), data['gt'].long().to(self.device)

                out = self.net(image)
                loss = self.criterion(out['y'], mask)

                if 'heatmap' in out:
                    heatmap = data['heatmap'].to(self.device)
                    m_loss = self.mse_loss(out['heatmap'], heatmap)
                    loss += 1.0 * m_loss
                    meters['mse'].update(m_loss.item(), image.size(0))

                diceEvalVal.addBatch(out['y'].max(1)[1], mask)
                t1 = time.time()

                meters['loss'].update(loss.item(), image.size(0))
                meters['time'].update(t1-t0)

                t0 = time.time()

        dice = diceEvalVal.getMetric()
        self.logger.info('Validate: Time=%.3fms/batch, Loss=%.6f, Mse=%.6f, Dice=%.4f' % \
                         (meters['time'].avg * 1000, meters['loss'].avg,
                          meters['mse'].avg, dice))

        return {'Dice': dice, 'Loss': meters['loss'].avg}        
    
    def validate_skeleton_subjects(self, stage='val'):
        
        val_path = cfg.TRAIN.DATA.VAL_LIST if stage == 'val' else cfg.TEST.DATA.TEST_FILE
        with open(val_path, 'r') as f:
            subjects = f.readlines()
            subjects = [subject.strip() for subject in subjects]
        subjects = sorted(subjects)
        self.logger.info("the number of subjects to be inferenced is %d" % len(subjects))        
        
        self.net.eval()
        meter_names = ['loss', 'time', 'hp_loss']
        meters = {name: AverageMeter() for name in meter_names}
        n_class = cfg.MODEL.PARA.NUM_CLASSES
        diceEvalVal = diceEval(n_class)
        
        heatmap = cfg.TRAIN.DATA.USE_HEATMAP if 'USE_HEATMAP' in cfg.TRAIN.DATA.keys() else False
        if heatmap:
            mse_loss = nn.SmoothL1Loss(reduction='mean').to(self.device)

        seg_results = []
        hp_results = []
        for subject in subjects:
            self.logger.info("start to process %s data" % subject)
            para_dict = {"subject": subject}
            test_set = VesselSubjectDS(para_dict, stage="val")
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
                                                      shuffle=False, pin_memory=True)
            with torch.no_grad():
                v_x, v_y, v_z = test_set.volume_size()
                seg_v = torch.FloatTensor(n_class, v_x, v_y, v_z).zero_().to(self.device)
                if heatmap:
                    num = torch.ByteTensor(v_x, v_y, v_z).zero_().to(self.device)
                    hp_v = torch.FloatTensor(v_x, v_y, v_z).zero_().to(self.device)
                
                for _, data in enumerate(test_loader):
                    img = data['img'].to(self.device)
                    msk = data['gt'].long().to(self.device)
                    coord = data['coord']
                    out = self.net(img)
                    pred = out['y']
                    loss = self.criterion(pred, msk)
                    meters['loss'].update(loss.item(), img.size(0))
                    diceEvalVal.addBatch(pred.max(1)[1], msk)
                    
                    if heatmap:
                        hp = data['hp'].to(self.device)
                        hp_loss = mse_loss(out['hp'], hp)
                        meters['hp_loss'].update(hp_loss.item(), img.size(0))
                    
                    pred = F.softmax(out['y'], dim=1)
                    for idx in range(img.size(0)):
                        sx, ex = coord[idx][0][0], coord[idx][0][1]
                        sy, ey = coord[idx][1][0], coord[idx][1][1]
                        sz, ez = coord[idx][2][0], coord[idx][2][1]
                        if stage == 'val':
                            seg_v[:, sx:ex, sy:ey, sz:ez] += pred[idx]
                        else:
                            seg_v[:, sx:ex, sy:ey, sz:ez] = torch.max(seg_v[:, sx:ex, sy:ey, sz:ez], pred[idx])
                        
                        if heatmap:
                            hp_v[sx:ex, sy:ey, sz:ez] += out['hp'][idx][0]
                            num[sx:ex, sy:ey, sz:ez] += 1
                
                if stage == 'val':
                    seg_v = seg_v.max(0)[1].byte().cpu().numpy()
                else:
                    va, vb = seg_v[1:, ...].max(0)
                    seg_v = (vb + 1).byte() * (va >= 0.5)
                    seg_v = seg_v.cpu().numpy()                   
                seg_results.append((seg_v, test_set.volume_mask()))
                
                if heatmap:
                    hp_v = (hp_v / num.float()).cpu().numpy()
                    l2 = pow(hp_v - test_set.volume_heatmap(), 2).mean()
                    gt = test_set.volume_mask()
                    dd = 2 * ((hp_v >= 1.0) * gt).sum() / ((hp_v >= 1.0).sum() + gt.sum())
                    hp_results.append([l2, dd])
                
                #save segment results
                if cfg.TEST.SAVE:
                    self.logger.info('save %s prediction results' % subject)
                    if stage == 'val':
                        test_set.save(seg_v, os.path.join(cfg.OUTPUT_DIR, '%s_seg.nii.gz' % subject))
                    else:
                        test_set.save(seg_v, os.path.join(cfg.TEST.SAVE_DIR, '%s_seg.nii.gz' % subject))
                        if heatmap:
                            test_set.save(hp_v, os.path.join(cfg.TEST.SAVE_DIR, '%s_hp.nii.gz' % subject))

        #evalutate seg results
        self.logger.info("evaluating segmentation results in async ...")
        ex = ProcessPoolExecutor()
        objs = []
        monitor = progress_monitor(total=len(subjects))
        #print('seg_results #########', seg_results)
        for (seg, gt) in seg_results:
            future  = ex.submit(eval_ccta_volume_v2, seg, gt, n_class, True)
            future.add_done_callback(fn=monitor)
            objs.append(future)
        
        ex.shutdown(wait=True)
        results = []
        for obj in objs:
            results.append(obj.result())        
        results = np.array(results)
        result = np.mean(results, 0)

        #print('results #########', results)
        
        if cfg.TEST.SAVE:
            np.save(os.path.join(cfg.OUTPUT_DIR if stage == 'val' else cfg.TEST.SAVE_DIR, 'eval_result.npy'), results)
        
        dd = diceEvalVal.getMetric()
        p_ss = 'class: (' + ','.join('%.4f' % d for d in dd[:-1]) + ')' + ' mean: %.4f, total: %.4f' % (dd[:-1].mean(), dd[-1])
        v_ss = 'class: (' + ','.join('%.4f' % d for d in result[:n_class-1]) + ')' + ' mean: %.4f, total: %.4f' % (result[:n_class-1].mean(), result[n_class-1])
        a_ss = 'class: (' + ','.join('%.4f' % d for d in result[n_class:2*n_class-1]) + ')'  + \
                  ' mean: %.4f, total: %.4f' % (result[n_class:2*n_class-1].mean(), result[2*n_class-1])
        
        names = ['VR2', 'VP2', 'SR', 'SP', 'SR1', 'SP1', 'SR2', 'SP2', 'F-SRp3SP', 'F-SRp5SP', 'F-SRp7SP']
        for i, subject in enumerate(subjects):
            sr, sp, sr1, sp1, sr2, sp2, vr2, vp2, clcr = results[i, -9:]  #sr sp = result[0],[1]  ,[2]
            
            sr3sp = 2 * (sr ** 3) * sp / (sr ** 3 + sp)  #SP = float(TN/float((TN+FP)+smooth)) 
            sr5sp = 2 * (sr ** 5) * sp / (sr ** 5 + sp)
            sr7sp = 2 * (sr ** 7) * sp / (sr ** 7 + sp)
            values = [vr2, vp2, sr, sp, sr1, sp1, sr2, sp2, sr3sp, sr5sp, sr7sp, clcr]
            self.logger.info('No.%03d/%03d %s: %s' % (i+1, len(subjects), subject, ', '.join(['%s=%.4f' % (name, value) for name, value in zip(names, values)])))
            
        sr, sp, sr1, sp1, sr2, sp2, vr2, vp2, clcr = np.mean(results[:, -9:], 0)
        
        sr3sp = 2 * (sr ** 3) * sp / (sr ** 3 + sp)
        sr5sp = 2 * (sr ** 5) * sp / (sr ** 5 + sp)
        sr7sp = 2 * (sr ** 7) * sp / (sr ** 7 + sp)
        values = [vr2, vp2, sr, sp, sr1, sp1, sr2, sp2, sr3sp, sr5sp, sr7sp, clcr]
        
        self.logger.info('Validate: loss=%.6f, patch_dice=[%s], volume_dice=[%s], assd=[%s], clcr=%.4f' % (meters['loss'].avg, p_ss, v_ss, a_ss, result[-1]))
        self.logger.info('            %s' % ', '.join(['%s=%.4f' % (name, value) for name, value in zip(names, values)]))
        
        ddict = {'Dice':dd[:n_class-1].mean(), 'VDice': result[n_class-1], 'Loss': -meters['loss'].avg, 'ASSD': -result[-2], 'CLCR': result[-1]}
        ddict['VDCLCR'] = result[n_class-1] * result[-1]
        ddict['ASCLCR'] = result[-1] / (result[-2] ** 0.1)
        for name, value in zip(names, values):
            ddict[name] = value
        
        #evaluate heatmap results
        if heatmap:
            hp_results = np.array(hp_results)
            hp_results = np.mean(hp_results, 0)
            self.logger.info('Validate heatmap: loss=%.6f, diff=%.4f, dice=%.4f' % (meters['hp_loss'].avg, hp_results[0], hp_results[1]))
            ddict['HL1'] = 1.0 - hp_results[0]
            ddict['HDice'] = hp_results[1]
        
        return ddict
                    
    def validate_subjects(self):
        
        with open(cfg.TRAIN.DATA.VAL_LIST, 'r') as f:
            subjects = f.readlines()
            subjects = [subject.strip() for subject in subjects]
            
        self.logger.info("the number of subjects to be inferenced is %d" % len(subjects))
        
        with open('/brain_data/dataset/ccta/cardiac_bbox.lst', 'r') as f:
            lines = f.readlines()
        ddict = {line.split(' ')[0]: [int(a) for a in line.split(' ')[1:]] for line in lines}
        
        self.net.eval()
        meter_names = ['loss', 'time']
        meters = {name: AverageMeter() for name in meter_names}
        
        seg_results = []
        for subject in subjects:            
            self.logger.info("start to process %s data" % subject)
            #para_dict = {"subject": subject, 'cardiac_bbox': ddict}
            para_dict = {"subject": subject}
            test_set = VesselDS(para_dict, stage="val")
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
                                                      shuffle=False, pin_memory=True)
            with torch.no_grad():
                v_x, v_y, v_z = test_set.volume_size()
                seg = torch.FloatTensor(v_x, v_y, v_z).zero_().to(self.device)
                num = torch.ByteTensor(v_x, v_y, v_z).zero_().to(self.device)
                for i, (image, coord) in enumerate(test_loader):
                    image = image.to(self.device)
                    out = self.net(image)
                    
                    #pred = out['y']
                    pred = F.softmax(out['y'], dim=1)
                    if cfg.TRAIN.DATA.USE_SDM and 'sdm' in out:
                        pred = (pred + torch.sigmoid(-1500*out['sdm'])) / 2
                    
                    for idx in range(image.size(0)):
                        sx, ex = coord[idx][0][0], coord[idx][0][1]
                        sy, ey = coord[idx][1][0], coord[idx][1][1]
                        sz, ez = coord[idx][2][0], coord[idx][2][1]
                        
                        seg[sx:ex, sy:ey, sz:ez] += pred[idx][1]
                        #seg[sx:ex, sy:ey, sz:ez] = torch.max(seg[sx:ex, sy:ey, sz:ez], pred[idx][1]
                        num[sx:ex, sy:ey, sz:ez] += 1
                        
                seg = ((seg / num.float()) >= 0.50).byte().cpu().numpy()
                #seg = (seg >= 0.50).byte().cpu().numpy()
                seg_results.append((seg, test_set.volume_mask()))
         
        #evalutate results
        self.logger.info("evaluating segmentation results in async ...")
        ex = ProcessPoolExecutor()
        objs = []
        monitor = progress_monitor(total=len(subjects))
        for (seg, gt) in seg_results:
            future  = ex.submit(eval_volume, seg, gt)    #####
            #future  = ex.submit(eval_xinji, seg, gt)
            future.add_done_callback(fn=monitor)
            objs.append(future)
        
        ex.shutdown(wait=True)
        results = []
        for obj in objs:
            results.append(obj.result())        
        results = np.array(results)
        result = np.mean(results, 0)
        #self.logger.info('Validate: dice=%.4f, d0=%.4f, d1=%.4f, d2=%.4f' % (np.mean(result), result[0], result[1], result[2]))
                         
        self.logger.info('Validate: dice=%.4f, hd=%.4f, assd=%.4f' % (result[0], result[1], result[2]))
        #self.logger.info('Validate: dice=%.4f, hd=%.4f, assd=%.4f, aorta dice=%.4f' % (result[0], result[1], result[2], result[3]))
        
        #return {'Dice': np.mean(result)}
        return {'Dice':result[0], 'HD':-result[1], 'ASSD':-result[2]}

    def test(self):
        #return self.validate_sdf(stage='test')
        return self.validate_skeleton_subjects(stage='test')
        
        subjects = []

        if os.path.exists(cfg.TEST.DATA.TEST_FILE):
            with open(cfg.TEST.DATA.TEST_FILE, 'r') as f:
                lines = f.readlines()
                subjects = [line.strip() for line in lines]
        else:
            subjects = cfg.TEST.DATA.TEST_LIST

        self.logger.info("the number of subjects to be inferenced is %d" % len(subjects))
        
        with open('/brain_data/dataset/ccta/cardiac_bbox.lst', 'r') as f:
            lines = f.readlines()
        ddict = {line.split(' ')[0]: [int(a) for a in line.split(' ')[1:]] for line in lines}
        
        scores = []
        messages = []

        self.net.eval()
        for subject in subjects:
            #para_dict = {"subject": subject}
            para_dict = {"subject": subject, 'cardiac_bbox': ddict}
            test_set = VesselDS(para_dict, stage="test")
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
                                                      shuffle=False, pin_memory=True)
            v_x, v_y, v_z = test_set.volume_size()
            seg = torch.FloatTensor(v_x, v_y, v_z).zero_().to(self.device)
            num = torch.ByteTensor(v_x, v_y, v_z).zero_().to(self.device)
            with torch.no_grad():
                for i, (image, coord) in enumerate(test_loader):
                    image = image.to(self.device)
                    out = self.net(image)
                    
                    pred = out['y']
                    pred = F.softmax(out['y'], dim=1)
                    #if cfg.TRAIN.DATA.USE_SDM and 'sdm' in out:
                    #    pred = (pred + torch.sigmoid(-1500*out['sdm'])) / 2
                        #pred = torch.max(pred, torch.sigmoid(-1500*out['sdm']))
                        
                    for idx in range(image.size(0)):
                        sx, ex = coord[idx][0][0], coord[idx][0][1]
                        sy, ey = coord[idx][1][0], coord[idx][1][1]
                        sz, ez = coord[idx][2][0], coord[idx][2][1]
                        
                        #seg[sx:ex, sy:ey, sz:ez] = torch.max(seg[sx:ex, sy:ey, sz:ez], pred[idx])
                        seg[sx:ex, sy:ey, sz:ez] += pred[idx][1]
                        num[sx:ex, sy:ey, sz:ez] += 1

            #seg = ((seg / num.float()) >= 0.50).byte().cpu().numpy()
            seg = (seg >= 0.50).byte().cpu().numpy()
            
            if cfg.TEST.SAVE:
                self.logger.info('save %s prediction results' % subject)
                subject = subject.split('.')[0].split('/')[-1]
                test_set.save(seg, os.path.join(cfg.TEST.SAVE_DIR, '%s_seg.nii.gz' % subject))
