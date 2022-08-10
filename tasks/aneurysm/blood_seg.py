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
from metrics.eval_vessel import eval_volume, eval_cta_volume
from .blood_loader import BloodDataset
from tasks.aneurysm.datasets.blood_seg import BLOOD_SEG, VesselDatasetSampler, data_prefetcher
from tasks.aneurysm.datasets.blood_dataset import VesselSubjectDS
from utils.config import cfg
from utils.tools.util import count_time, progress_monitor
from tasks.aneurysm.nets.vessel_net import SimpleNet, DASEResNet34, DASEResNet18, DAMultiHeadResNet
from utils.tools.util import count_time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from loss.modules.loss_modules import *

from tasks.aneurysm.datasets.base_dataset import RandomSequenceSampler

def worker_init_fn(worker_id):
    np.random.seed(cfg.SEED)

class BloodSeg(Task):

    def __init__(self):
        super(BloodSeg, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        '''
        if cfg.TASK.STATUS == 'train':
            self.online_sample = 'subjects' in cfg.TRAIN.DATA.TRAIN_LIST
            if self.online_sample:
                self.data_sampler = VesselDatasetSampler(cfg.TRAIN.DATA.TRAIN_LIST)
                self.data_sampler.asyn_sample(2000, 10)
            else:
                self.train_set = BLOOD_SEG({"train_list":cfg.TRAIN.DATA.TRAIN_LIST}, stage="train")
            
            self.val_subjects = 'subjects' in cfg.TRAIN.DATA.VAL_LIST
            if not self.val_subjects:
                para_dict = {"val_list": cfg.TRAIN.DATA.VAL_LIST}
                val_set = BLOOD_SEG(para_dict, stage="val")
                self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
                                                              shuffle=False, pin_memory=True)

                self.logger.info('Total validate data: %d' % len(self.val_loader))
          '''
        self.bdice = BinaryDiceLoss().to(self.device)
        self.sdm = SDMLoss().to(self.device)
    
    def get_model(self):
        if cfg.MODEL.NAME == 'simple_net':
            self.net = SimpleNet(cfg.MODEL.NCLASS, k=16)
        elif cfg.MODEL.NAME == 'da_seresnet34':
            self.net = DASEResNet34(cfg.MODEL.NCLASS, k=cfg.MODEL.K, input_channels=cfg.MODEL.INPUT_CHANNEL, drop_rate=0.0)
        elif cfg.MODEL.NAME == 'da_seresnet18':
            self.net = DASEResNet18(cfg.MODEL.NCLASS, k=cfg.MODEL.K, input_channels=cfg.MODEL.INPUT_CHANNEL,
                                   heatmap= cfg.TRAIN.DATA.USE_HEATMAP if 'USE_HEATMAP' in cfg.TRAIN.DATA.keys() else False)
        elif cfg.MODEL.NAME == 'da_seresnet18_mh':
            self.net = DASEResNet18_MH(cfg.MODEL.NCLASS, k=cfg.MODEL.K, input_channels=cfg.MODEL.INPUT_CHANNEL)
        elif cfg.MODEL.NAME == 'da_multihead_resnet':
            self.net = DAMultiHeadResNet(cfg.MODEL.NCLASS, k=cfg.MODEL.K, input_channels=cfg.MODEL.INPUT_CHANNEL)
        else:
            super().get_model()
    
    def train_npz(self, epoch):
        
        self.net.train()
        meter_names = ['time', 'loss', 'seg', 'boundary', 'sdm', 'bdice']
        meters = {name: AverageMeter() for name in meter_names}
        #diceEvalTrain = diceEval(cfg.MODEL.NCLASS)
        #diceSDMEvalTrain = diceEval(cfg.MODEL.NCLASS)
        
        if self.online_sample:
            self.train_set = self.data_sampler.get_data_loader()
            if self.data_sampler.memory and epcoh < cfg.SOLVER.EPOCHS-1:
                self.data_sampler.asyn_sample(500, 10)
            
        kwargs = {'worker_init_fn': worker_init_fn, 'num_workers': cfg.TRAIN.DATA.WORKERS,
                  'pin_memory': True, 'drop_last': True}

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=cfg.TRAIN.DATA.BATCH_SIZE,
                                                        shuffle=True, **kwargs)
        self.logger.info('epoch=%04d, total train data: %d' % (epoch, len(self.train_loader)))
        # epoch 模式更新学习率
        self.lr_scheduler.step()
        self.logger.info("current epoch learning rate:{:.8f}!".format(self.lr_scheduler.get_lr()[0]))
        
        t0 = time.time()
        #for batch_idx, data in enumerate(self.train_loader):
        #    image, mask = data['img'].to(self.device), data['gt'].long().to(self.device)       
        prefetch_train_loader = data_prefetcher(self.train_loader)
        print('\n****** using prefetcher for train data ******\n')
        for batch_idx, data in enumerate(prefetch_train_loader):
            
            image, mask  = data['img'], data['gt']  
            import pdb
            pdb.set_trace()
            self.optimizer.zero_grad()
            out = self.net(image)
            seg = out['y']
            loss = 0
            seg_loss = self.criterion(seg, mask)    
            loss +=  seg_loss
            meters['seg'].update(seg_loss.item(), image.size(0))
            
            if 'sdm' in data:
                sdm = data['sdm']
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
                #diceEvalTrain.addBatch(seg.max(1)[1], mask)
                diceEvalTrain.addBatch((seg >= 0.5).long(), mask)
                if 'sdm' in out:
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
        
        if self.online_sample:
            if not self.data_sampler.memory and epoch < cfg.SOLVER.EPOCHS-1:
                self.data_sampler.asyn_sample(2000, 10)
    
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
        #self.lr_scheduler.step()
        self.logger.info("current epoch learning rate:{:.8f}!".format(self.lr_scheduler.get_lr()[0]))
        
        heatmap = cfg.TRAIN.DATA.USE_HEATMAP if 'USE_HEATMAP' in cfg.TRAIN.DATA.keys() else False
        if heatmap:
            mse_loss = nn.SmoothL1Loss(reduction='mean').to(self.device)
        skeleton = cfg.TRAIN.DATA.USE_SKELETON if 'USE_SKELETON' in cfg.TRAIN.DATA.keys() else False
        if skeleton:
            ce_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 10.0]).cuda(), ignore_index=255)
            
        for batch_idx, data in enumerate(train_loader):

            image, mask = data['img'].to(self.device), data['gt'].long().to(self.device)
            out = self.net(image)            
            seg = out['y']
            loss = 0
            seg_loss = self.criterion(seg, mask)    
            loss +=  seg_loss
            
            if skeleton:
                skel = data['skel'].long().to(self.device)
                skel_loss = ce_loss(seg, skel)
                loss += 0.5*skel_loss
                meters['skel_loss'].update(skel_loss.item(), image.size(0))
            
            if heatmap:
                hp = data['hp'].to(self.device)
                hp_loss = mse_loss(out['hp'], hp)
                loss += hp_loss
                meters['hp_loss'].update(hp_loss.item(), image.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            import pdb
            pdb.set_trace()
            if batch_idx % 3 == 0:
                diceEvalTrain.addBatch(seg.max(1)[1], mask)

            t1 = time.time()
            meters['time'].update(t1-t0)
            meters['loss'].update(loss.item(), image.size(0))

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
    def train(self, epoch):
        self.train_patches(epoch)
                
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
                                                                      shuffle=False, pin_memory=True, drop_last=False)
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
            
        #evalutate results
        self.logger.info("evaluating segmentation results in async ...")
        ex = ProcessPoolExecutor()
        objs = []
        monitor = progress_monitor(total=len(subjects))
        for (seg, gt) in seg_results:
            future  = ex.submit(eval_cta_volume, seg, gt, True)
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
        return self.validate_skeleton_subjects(stage='test')
    
        subjects = []        
        if os.path.exists(cfg.TEST.DATA.TEST_FILE):
            with open(cfg.TEST.DATA.TEST_FILE, 'r') as f:
                lines = f.readlines()
                subjects = [line.strip() for line in lines]
        else:
            subjects = cfg.TEST.DATA.TEST_LIST
        
        self.logger.info("the number of subjects to be inferenced is %d" % len(subjects))
        scores = []
        messages = []
        
        self.net.eval()
        for subject in subjects:
            test_set = BLOOD_SEG( {"subject": subject}, stage="test")
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
                                                      shuffle=False, pin_memory=True)
            v_x, v_y, v_z = test_set.volume_size()            
            seg = torch.FloatTensor(v_x, v_y, v_z).zero_().to(self.device)
            num = torch.ByteTensor(v_x, v_y, v_z).zero_().to(self.device)
            
            with torch.no_grad():
                for i, (image, coord) in enumerate(test_loader):
                    image = image.to(self.device)
                    out = self.net(image)
                
                    #pred = F.softmax(out['y'], dim=1)
                    pred = out['y']
                    if 'sdm' in out:
                        pred = (pred + torch.sigmoid(-1500*out['sdm'])) / 2
                        
                    for idx in range(image.size(0)):
                        sx, ex = coord[idx][0][0], coord[idx][0][1]
                        sy, ey = coord[idx][1][0], coord[idx][1][1]
                        sz, ez = coord[idx][2][0], coord[idx][2][1]
                        
                        #seg[sx:ex, sy:ey, sz:ez] = torch.max(seg[sx:ex, sy:ey, sz:ez], pred[idx][1])
                        seg[sx:ex, sy:ey, sz:ez] += pred[idx]
                        num[sx:ex, sy:ey, sz:ez] += 1
            
            #seg = (seg >= 0.50).byte().cpu().numpy()        
            seg = ((seg / num.float()) >= 0.50).byte().cpu().numpy()    
            
            if cfg.TEST.SAVE:
                self.logger.info('save %s prediction results' % subject)
                subject = subject.split('.')[0].split('/')[-1]
                test_set.save(seg, os.path.join(cfg.TEST.SAVE_DIR, '%s_seg.nii.gz' % subject))
        
    def test2(self):

        subjects = []

        if os.path.exists(cfg.TEST.DATA.TEST_FILE):
            with open(cfg.TEST.DATA.TEST_FILE, 'r') as f:
                lines = f.readlines()
                subjects = [line.strip() for line in lines]
        else:
            subjects = cfg.TEST.DATA.TEST_LIST

        self.logger.info("the number of subjects to be inferenced is %d" % len(subjects))
        scores = []
        messages = []

        self.net.eval()
        for subject in subjects:
            para_dict = {"subject": subject}
            test_set = BLOOD_SEG(para_dict, stage="test")
            # test_set = Volume3DLoader(subject)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
                                                      shuffle=False, pin_memory=True)
            v_x, v_y, v_z = test_set.volume_size()
            seg = torch.FloatTensor(2, v_x, v_y, v_z).zero_().to(self.device)
            heatmap = torch.FloatTensor(v_x, v_y, v_z).zero_().to(self.device)
            num = np.zeros((2, v_x, v_y, v_z), 'uint8')

            with torch.no_grad():
                for i, (image, coord) in enumerate(test_loader):
                    image = image.to(self.device)
                    out = self.net(image)

                    pred = F.softmax(out['y'], dim=1)
                    for idx in range(image.size(0)):
                        sx, ex = coord[idx][0][0], coord[idx][0][1]
                        sy, ey = coord[idx][1][0], coord[idx][1][1]
                        sz, ez = coord[idx][2][0], coord[idx][2][1]

                        seg[:, sx:ex, sy:ey, sz:ez] += pred[idx]
                        heatmap[sx:ex, sy:ey, sz:ez] += out['heatmap'][idx][0]
                        num[:, sx:ex, sy:ey, sz:ez] += 1

            if cfg.TEST.SAVE:
                #prob = seg.cpu().numpy() / num.astype('float32')
                #test_set.save(prob[0,...], os.path.join(cfg.TEST.SAVE_DIR, '%s_bg.nii.gz' % subject), True)

                hp = heatmap.cpu().numpy() / num.astype('float32')[0, ...]
                test_set.save(hp, os.path.join(cfg.TEST.SAVE_DIR, '%s_heatmap.nii.gz' % subject), True)


            seg = seg.max(0)[1].cpu().numpy()
            #seg = seg[1, :, :, :]
            #seg = (seg >= 0.50).byte().cpu().numpy()

            labels = measure.label(seg)
            vv = np.unique(labels)
            for i in range(1, len(vv)):
                mm = labels == i
                if mm.sum() < 100:
                    seg[mm] = 0
                else:
                    bb = np.where(mm)
                    x0, x1 = np.min(bb[0]), np.max(bb[0])
                    if x1 + 1 - x0 < 5:
                        seg[mm] = 0

            if test_set.get_gt() is not None:
                seg_ = torch.from_numpy(seg).cuda()
                gt_ = torch.from_numpy(test_set.get_gt()).cuda()
                score, _ = tensor_dice(seg_, gt_, cfg.MODEL.NCLASS)
                scores.append(score)
                msg =  '%s dice: %.4f' % (subject, score)
                messages.append(msg)
                self.logger.info(msg)

            if cfg.TEST.SAVE:
                self.logger.info('save %s prediction results' % subject)
                subject = subject.split('.')[0].split('/')[-1]
                test_set.save(seg, os.path.join(cfg.TEST.SAVE_DIR, '%s_seg.nii.gz' % subject))

        self.logger.info('mean: %.4f, var: %.4f' % (np.mean(scores), np.std(scores)))

        if cfg.TEST.SAVE:
            with open(os.path.join(cfg.TEST.SAVE_DIR, 'dice.txt'), 'w') as f:
                for msg in messages:
                    f.write(msg + '\n')
                f.write('mean: %.4f, var: %.4f' % (np.mean(scores), np.std(scores)))
