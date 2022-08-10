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

from .dataset import VesselSubjectDS
from utils.config import cfg
from tasks.aneurysm.nets.resunet import DAResUNet
from tasks.aneurysm.nets.vessel_net import DASEResNet18
from tasks.intracranial_vessel.nets.head_net import HeadNet4, SEHeadNet4, HeadNet3, SEHeadNet3
from utils.tools.util import count_time, progress_monitor
from metrics.eval_vessel import eval_volume, eval_xinji, eval_ccta_volume
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from loss.modules.loss_modules import *
from tasks.aneurysm.datasets.base_dataset import RandomSequenceSampler

def worker_init_fn(worker_id):
    np.random.seed(cfg.SEED)
    
class INTRAVesselSeg(Task):
    
    def __init__(self):
        super(INTRAVesselSeg, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    def get_model(self):
        if cfg.MODEL.NAME == 'da_resunet':
            self.net = DAResUNet(cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'da_seresnet18':
            self.net = DASEResNet18(
                cfg.MODEL.PARA.NUM_CLASSES,
                k=24,
                input_channels=1 if cfg.TRAIN.DATA.USE_CASCADE is False else 2,
                heatmap=cfg.TRAIN.DATA.USE_HEATMAP if 'USE_HEATMAP' in cfg.TRAIN.DATA.keys() else False
            )
        elif cfg.MODEL.NAME == 'head_net3':
            self.net = HeadNet3(num_classes=cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K)
        elif cfg.MODEL.NAME == 'sehead_net3':
            self.net = SEHeadNet3(num_classes=cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K, attention=False)
        elif cfg.MODEL.NAME == 'dasehead_net3':
            self.net = SEHeadNet3(num_classes=cfg.MODEL.PARA.NUM_CLASSES, k=cfg.MODEL.K, attention=True)
        else:
            super().get_model()
    
    @count_time
    def train(self, epoch):
        self.train_patches(epoch)
    
    def train_patches(self, epoch):
        
        with open(cfg.TRAIN.DATA.TRAIN_LIST, 'r') as f:
            subjects = f.readlines()
            subjects = [subject.strip() for subject in subjects]
        
        batch_size = cfg.TRAIN.DATA.BATCH_SIZE
        sample_per_subject = batch_size * 1
        train_set = VesselSubjectDS({"subjects":subjects, "sample": sample_per_subject}, stage="train")
        
        self.net.train()
        meter_names = ['time', 'loss', 'skel_loss', 'hp_loss']
        meters = {name: AverageMeter() for name in meter_names}
        diceEvalTrain = diceEval(cfg.MODEL.PARA.NUM_CLASSES)
        
        kwargs = {'batch_size': batch_size, 'shuffle': False,
                      'num_workers': cfg.TRAIN.DATA.WORKERS, #'worker_init_fn': worker_init_fn, 
                      'pin_memory': True, 'drop_last': True,
                      'sampler': RandomSequenceSampler(len(subjects), sample_per_subject, batch_size, cfg.TRAIN.DATA.WORKERS, 
                                                       max_subject_in_batch = 8)}
        train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
        self.logger.info('epoch=%04d, total train data: %d' % (epoch, len(train_loader)))
        t0 = time.time()
        self.logger.info("current epoch learning rate:{:.8f}!".format(self.lr_scheduler.get_lr()[0]))
        
        heatmap = cfg.TRAIN.DATA.USE_HEATMAP if 'USE_HEATMAP' in cfg.TRAIN.DATA.keys() else False
        if heatmap:
            mse_loss = nn.SmoothL1Loss(reduction='mean').to(self.device)
        skeleton = cfg.TRAIN.DATA.USE_SKELETON if 'USE_SKELETON' in cfg.TRAIN.DATA.keys() else False
        if skeleton:
            ce_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 10.0]).cuda(), ignore_index=255)
            
        for batch_idx, data in enumerate(train_loader):

            image, mask = data['img'].to(self.device), data['gt'].long().to(self.device)
            if cfg.TRAIN.DATA.USE_CASCADE is True:
                feature = data['feature'].to(self.device)
                image = torch.cat((image,feature), dim=1)
                # from IPython import embed; embed()

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
            
            if skeleton and epoch >= 30:
                skel = data['skel'].long().to(self.device)
                skel_loss = ce_loss(seg, skel)
                loss += 0.5*skel_loss
                meters['skel_loss'].update(0.5*skel_loss.item(), image.size(0))
            
            if heatmap:
                hp = data['hp'].to(self.device)
                hp_loss = mse_loss(out['hp'], hp)
                loss += hp_loss
                meters['hp_loss'].update(hp_loss.item(), image.size(0))
            
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
        return self.validate_subjects()
    
    def validate_subjects(self, stage='val'):
        
        val_path = cfg.TRAIN.DATA.VAL_LIST if stage == 'val' else cfg.TEST.DATA.TEST_FILE
        with open(val_path, 'r') as f:
            subjects = f.readlines()
            subjects = [subject.strip() for subject in subjects]

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

                    if cfg.TRAIN.DATA.USE_CASCADE is True:
                        feature = data['feature'].to(self.device)
                        img = torch.cat((img, feature), dim=1)

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
        for (seg, gt) in seg_results:
            future  = ex.submit(eval_volume, seg, gt, n_class, True)
            future.add_done_callback(fn=monitor)
            objs.append(future)
        
        ex.shutdown(wait=True)
        results = []
        for obj in objs:
            results.append(obj.result())        
        results = np.array(results)
        result = np.mean(results, 0)
        
        dd = diceEvalVal.getMetric()
        p_ss = 'class: (' + ','.join('%.4f' % d for d in dd[:-1]) + ')' + ' mean: %.4f, total: %.4f' % (dd[:-1].mean(), dd[-1])
        v_ss = 'class: (' + ','.join('%.4f' % d for d in result[:n_class-1]) + ')' + ' mean: %.4f, total: %.4f' % (result[:n_class-1].mean(), result[n_class-1])
        self.logger.info('Validate: loss=%.6f, patch_dice=[%s], volume_dice=[%s], hd=%.4f, assd=%.4f, clcr=%.4f' % (meters['loss'].avg, p_ss, v_ss, result[-3], result[-2], result[-1]))
        
        ddict = {'Dice':dd[:n_class-1].mean(), 'VDice': result[n_class-1], 'Loss': -meters['loss'].avg, 'HD':-result[-3], 'ASSD': -result[-2], 'CLCR': result[-1]}
        ddict['VDCLCR'] = result[n_class-1] * result[-1]
        
        #evaluate heatmap results
        if heatmap:
            hp_results = np.array(hp_results)
            hp_results = np.mean(hp_results, 0)
            self.logger.info('Validate heatmap: loss=%.6f, diff=%.4f, dice=%.4f' % (meters['hp_loss'].avg, hp_results[0], hp_results[1]))
            ddict['HL1'] = 1.0 - hp_results[0]
            ddict['HDice'] = hp_results[0]
        
        return ddict
    
    def test(self):
        
        return self.validate_subjects(stage='test')