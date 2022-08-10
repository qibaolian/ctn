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
from tasks.aneurysm.prefetcher import data_prefetcher

import threading

def worker_init_fn(worker_id):
    np.random.seed(cfg.SEED)


class AneurysmSeg(Task):

    def __init__(self):
        super(AneurysmSeg, self).__init__()

        if cfg.TASK.STATUS == 'train':

            self.data_sampler = AneurysmDataset(cfg.TRAIN.DATA.TRAIN_LIST)
            self.data_sampler.asyn_sample(50, 100, max_workers=25)
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
            super(AneurysmSeg, self).get_model()

    @count_time
    def train(self, epoch):

        self.net.train()
        meter_names = ['loss', 'time']
        meters = {name: AverageMeter() for name in meter_names}

        diceEvalTrain = diceEval(cfg.MODEL.NCLASS)
        aucEvalTrain = aucEval(cfg.MODEL.NCLASS)

        #train_set = self.data_sampler.sample(2, 100)
        train_set = self.data_sampler.get_data_loader()
        self.data_sampler.asyn_sample(300, 50)  # 50, 100
        # self.data_sampler.asyn_sample(1, 2)  # for fast train

        kwargs = {'worker_init_fn': worker_init_fn, 'num_workers': cfg.TRAIN.DATA.WORKERS,
                  'pin_memory': True, 'drop_last': True}

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.TRAIN.DATA.BATCH_SIZE,
                                                   shuffle=True, **kwargs)
        self.logger.info('epoch=%04d, total train data: %d' % (epoch, len(self.train_loader)))

        t0 = time.time()
        # epoch 模式更新学习率
        self.lr_scheduler.step()
        self.logger.info("current epoch learning rate:{:.8f}!".format(self.lr_scheduler.get_lr()[0]))

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # for batch_idx, data in enumerate(self.train_loader):
        #
        #     image, mask = data['img'].to(device), data['gt'].long().to(device)
        self.prefetch_train_loader = data_prefetcher(self.train_loader)
        print('\n****** using prefetcher for train data ******')
        for batch_idx, (image, mask) in enumerate(self.prefetch_train_loader):

            self.optimizer.zero_grad()
            out = self.net(image)
            seg = out['y']

            loss = 0
            if isinstance(seg, tuple):
                weights = [1.0, 0.8, 0.4, 0.2, 0.1]
                for i in range(len(out)):
                    loss += weights[i] * self.criterion(seg[i], mask)

                seg = seg[0]
            else:
                loss = self.criterion(seg, mask)

            loss.backward()
            self.optimizer.step()

            if batch_idx % 5 == 0:
                diceEvalTrain.addBatch(seg.max(1)[1], mask)

            t1 = time.time()
            meters['time'].update(t1-t0)
            meters['loss'].update(loss.item(), image.size(0))

            #if not ds:
            #    if batch_idx % 10 == 0:
            #        diceEvalTrain.addBatch(out.max(1)[1].data, mask.data)
            #else:
            #    aucEvalTrain.addBatch(out_cls.max(1)[1].data, target.data)
            #    diceEvalTrain.addBatch(out_seg.max(1)[1].data, mask.data)



            if batch_idx % cfg.TRAIN.PRINT == 0:
                dice = diceEvalTrain.getMetric()
                self.logger.info('epoch=%03d, batch_idx=%04d, Time=%.2fs, loss=%.4f, dice=%.4f' % \
                                 (epoch, batch_idx, meters['time'].avg, meters['loss'].avg, dice))

            t0 = time.time()
        torch.cuda.empty_cache()

    @count_time
    def validate(self):
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.net.eval()
        meter_names = ['loss', 'time']
        meters = {name: AverageMeter() for name in meter_names}
        diceEvalVal = diceEval(cfg.MODEL.NCLASS)
        aucEvalVal = aucEval(cfg.MODEL.NCLASS)
        
        with torch.no_grad():
            t0 = time.time()
            for i, data in enumerate(self.val_loader):
                
                image, mask = data['img'].to(device), data['gt'].long().to(device)
                out = self.net(image)
                loss = self.criterion(out['y'], mask)
                
                diceEvalVal.addBatch(out['y'].max(1)[1], mask)
                t1 = time.time()
                
                meters['loss'].update(loss.item(), image.size(0))
                meters['time'].update(t1-t0)
            
                t0 = time.time()
        
        
        dice = diceEvalVal.getMetric()
        self.logger.info('Validate: Time=%.2fms/batch, Loss=%.4f, Dice=%.4f' % \
                         (meters['time'].avg * 1000, meters['loss'].avg, dice))
        
        
        return {'Dice': dice, 'Loss': meters['loss'].avg}
        '''

        return self.validate_patches()

    def validate_patches(self):

        #val_set = ValidatePatch3DLoader(cfg.TRAIN.DATA.VAL_LIST)
        #val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
        #                                                         shuffle=False, pin_memory=True)
        meter_names = ['loss', 'time']
        meters = {name: AverageMeter() for name in meter_names}

        self.net.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info('start to validate cuda operation')

        pred_list, gt_list = [], []
        with torch.no_grad():

            t0 = time.time()
            ts = time.time()
            img_tensor, gt_tensor, _ = self.val_loader.get_tensor()
            batch_size=cfg.TEST.DATA.BATCH_SIZE
            for i in range(0, img_tensor.size(0), batch_size):
            #for i, data in enumerate(val_loader):

                t0 = time.time()
                #image, mask = data['img'].to(device), data['gt'].long().to(device)
                image, mask = img_tensor[i:i+batch_size].to(device), gt_tensor[i:i+batch_size].long().to(device)
                out = self.net(image)
                loss = self.criterion(out['y'], mask)
                t1 = time.time()
                meters['loss'].update(loss.item(), image.size(0))
                meters['time'].update(t1-t0)

                pred_list.append(out['y'].max(1)[1].cpu().numpy())
                #gt_list.append(data['gt'].numpy())
                print('\r GPU Process: %d/%d, time=%.3f s' % (i, img_tensor.size(0), time.time() - ts), end='')

        torch.cuda.empty_cache()
        print('\n GPU Process finished, and cost time=%.3f s' % (time.time() - ts,))
        #pred, gt = np.concatenate(pred_list), np.concatenate(gt_list)
        pred, gt = np.concatenate(pred_list), gt_tensor.numpy()
        self.logger.info('end to validate cuda operation, Time=%.2fms/batch' % (meters['time'].avg * 1000))

        param_list = [(pred[i:i+200], gt[i:i+200]) for i in range(0, len(pred), 200)]
        results = eval_results(param_list, False)

        dice_total = 0
        infarct_region_recall, infarct_region_recall_total = 0, 0
        infarct_region_precision, infarct_region_pred_total = 0, 0
        correct, total  = 0, 0
        for result in results:
            dice,  n_recall, n_gt, n_precison, n_predict, n_correct, n_total = result
            if n_gt > 0:
                dice_total += dice
            infarct_region_recall += n_recall
            infarct_region_recall_total += n_gt
            infarct_region_precision += n_precison
            infarct_region_pred_total += n_predict
            correct += n_correct
            total += n_total

        dice = dice_total / infarct_region_recall_total
        acc = correct / total

        precision = infarct_region_precision / (infarct_region_pred_total + 1e-4)
        recall = infarct_region_recall / (infarct_region_recall_total +1e-4)
        f1_score = 2.0 * (recall * precision) / (recall + precision + 1e-4)

        self.logger.info(
            'In total: Acc=%.4f (%d/%d), F1-score=%.4f, Precision=%.4f (%d/%d), Recall=%.4f (%d/%d),  Dice=%.4f, Loss=%.4f' % \
            (acc, correct, total,f1_score,
             precision, infarct_region_precision, infarct_region_pred_total,
             recall, infarct_region_recall, infarct_region_recall_total,
             dice, meters['loss'].avg)
        )

        return {
            'Loss': meters['loss'].avg,
            'Dice': dice,
            'F1-Score': f1_score,
            'Recall': recall,
            'Precision': precision,
        }

    def validate_subjects(self):

        with open(cfg.TRAIN.DATA.VAL_LIST, 'r') as f:
            subjects = f.readlines()
            subjects = [subject.strip() for subject in subjects]
        subjects = subjects[:2]
        self.logger.info("the number of subjects to be inferenced is %d" % len(subjects))

        self.net.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meter_names = ['loss', 'time']
        meters = {name: AverageMeter() for name in meter_names}

        seg_results = []
        for subject in subjects:
            self.logger.info("start to process %s data" % subject)
            para_dict = {"subject": subject}
            test_set = ANEURYSM_SEG(para_dict, "test")
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
                                                      shuffle=False, pin_memory=True)

            v_x, v_y, v_z = test_set.volume_size()
            with torch.no_grad():
                seg = torch.FloatTensor(2, v_x, v_y, v_z).zero_()
                seg = seg.to(device)
                for i, (image, mask, coord) in enumerate(test_loader):
                    image, mask = image.to(device), mask.long().to(device)
                    out = self.net(image)
                    loss = self.criterion(out['y'], mask)
                    meters['loss'].update(loss.item(), image.size(0))

                    pred = F.softmax(out['y'], dim=1)
                    for idx in range(image.size(0)):
                        sx, ex = coord[idx][0][0], coord[idx][0][1]
                        sy, ey = coord[idx][1][0], coord[idx][1][1]
                        sz, ez = coord[idx][2][0], coord[idx][2][1]

                        seg[:, sx:ex, sy:ey, sz:ez] += pred[idx]

                seg = seg[1, :, :, :]
                seg = (seg >= 0.50).byte().cpu().numpy()

                seg_results.append((seg, test_set.get_gt()))

            self.logger.info("finish process %s data" % subject)

        results = eval_results(seg_results)

        dice_list = []
        infarct_recall, infarct_total = 0, 0
        infarct_region_recall, infarct_region_recall_total = 0, 0
        infarct_region_precision, infarct_region_pred_total = 0, 0
        health_recall, health_total = 0, 0
        fp_num = 0
        for (subject, result) in zip(subjects, results):

            dice,  n_recall, n_gt, n_precison, n_predict = result

            if n_gt > 0:
                dice_list.append(dice)
                infarct_total += 1

            if n_recall > 0:
                infarct_recall += 1

            infarct_region_recall += n_recall
            infarct_region_recall_total += n_gt

            infarct_region_precision += n_precison
            infarct_region_pred_total += n_predict

            if n_gt == 0 and n_predict == 0:
                health_recall += 1

            if n_gt == 0:
                health_total += 1

            fp_num = n_predict - n_precison

            self.logger.info('%s: Recall=%d/%d, Precision=%d/%d, FP=%d, Dice=%.4f' % \
                                 (subject, n_recall, n_gt, n_precison, n_predict, n_predict - n_precison, dice))

        precision = infarct_region_precision / (infarct_region_pred_total + 1e-4)
        recall = infarct_region_recall / (infarct_region_recall_total +1e-4)
        sensitivity = infarct_recall / ( infarct_total + 1e-4)
        specificity =  health_recall / (health_total + 1e-4)
        dice = np.mean(dice_list)

        self.logger.info('In total: Preciison=%.4f, Recall=%.4f, Sensitivity=%.4f, Specificity=%.4f, Dice=%.4f' % \
                             (precision, recall, sensitivity, specificity, dice))

        return {'Sensitivity' : sensitivity,
                  'Specificity' : specificity,
                  'Dice': dice,
                  'F1-Score':   2.0*(recall * precision) / (recall + precision + 1e-4) }

    @count_time
    def test(self):
        subjects = []

        if os.path.exists(cfg.TEST.DATA.TEST_FILE):
            with open(cfg.TEST.DATA.TEST_FILE, 'r') as f:
                lines = f.readlines()
                subjects = [line.strip() for line in lines]
        else:
            subjects = cfg.TEST.DATA.TEST_LIST
        self.logger.info("the number of subjects to be inferenced is %d" % len(subjects))

        self.net.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for step, subject in enumerate(subjects, start=1):
            para_dict = {"subject": subject}
            test_set = ANEURYSM_SEG(para_dict, "test")
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
                                                      shuffle=False, pin_memory=True)
            #t0 = time.time()
            v_x, v_y, v_z = test_set.volume_size()

            with torch.no_grad():

                seg = torch.FloatTensor(v_x, v_y, v_z).zero_()
                seg = seg.to(device)

                for i, (image, coord) in enumerate(test_loader):
                    image = image.to(device)
                    out = self.net(image)

                    pred = F.softmax(out['y'], dim=1)
                    for idx in range(image.size(0)):
                        sx, ex = coord[idx][0][0], coord[idx][0][1]
                        sy, ey = coord[idx][1][0], coord[idx][1][1]
                        sz, ez = coord[idx][2][0], coord[idx][2][1]

                        seg[sx:ex, sy:ey, sz:ez] += pred[idx][1]
                        #seg[sx:ex, sy:ey, sz:ez] += pred[idx].max(0)[1]

                seg = (seg >= 0.30).cpu().numpy().astype(np.uint8)  # mask 0/1
                # seg = seg.cpu().numpy().astype(np.float32) # prob matrix [0, 1]

            if cfg.TEST.SAVE:
                test_set.save(seg, cfg.TEST.SAVE_DIR)

            # self.logger.info('finish process %s' % subject)
            self.logger.info('%d/%d: %s finished!' % (step, len(subjects), subject))

    def test2(self):

        subjects = []

        if os.path.exists(cfg.TEST.DATA.TEST_FILE):
            with open(cfg.TEST.DATA.TEST_FILE, 'r') as f:
                lines = f.readlines()
                subjects = [line.strip() for line in lines]
        else:
            subjects = cfg.TEST.DATA.TEST_LIST
        self.logger.info("the number of subjects to be inferenced is %d" % len(subjects))
        messages = []
        scores_1, scores_2 = [], []
        recalls_1, recalls_2 = [], []
        fps_1, fps_2 = [], []

        self.net.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for subject in subjects:
            para_dict = {"subject": subject}
            test_set = ANEURYSM_SEG(para_dict, "test")
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.DATA.BATCH_SIZE,
                                                      shuffle=False, pin_memory=True)
            #t0 = time.time()
            v_x, v_y, v_z = test_set.volume_size()

            with torch.no_grad():

                seg = torch.FloatTensor(2, v_x, v_y, v_z).zero_()
                seg = seg.to(device)

                #seg = torch.LongTensor(v_x, v_y, v_z).zero_()
                #seg = seg.cuda()

                for i, (image, coord) in enumerate(test_loader):
                    image = image.to(device)
                    out = self.net(image)

                    pred = F.softmax(out['y'], dim=1)
                    for idx in range(image.size(0)):
                        sx, ex = coord[idx][0][0], coord[idx][0][1]
                        sy, ey = coord[idx][1][0], coord[idx][1][1]
                        sz, ez = coord[idx][2][0], coord[idx][2][1]

                        seg[:, sx:ex, sy:ey, sz:ez] += pred[idx]
                        #seg[sx:ex, sy:ey, sz:ez] += pred[idx].max(0)[1]


                #seg = seg.max(0)[1]
                #print 'cuda seg %.4f' % (time.time() - t0)
                #t0 = time.time()
                seg = seg[1, :, :, :]
                seg = (seg >= 0.50).byte()

                #seg = (seg >= 1).byte()

                gt = torch.from_numpy(test_set.get_gt()).to(device)
                mask = torch.from_numpy(test_set.get_mask()).to(device)

            score_1, _ = tensor_dice(seg, gt, cfg.MODEL.NCLASS)
            scores_1.append(score_1)

            seg_2 = seg * mask
            score_2, _ = tensor_dice(seg_2, gt, cfg.MODEL.NCLASS)
            scores_2.append(score_2)
            #print 'dice %.4f' % (time.time() - t0)
            #t0 = time.time()

            seg = seg.cpu().numpy()
            seg_2 = seg_2.cpu().numpy()
            mask = mask.cpu().numpy()
            gt = gt.cpu().numpy()

            recall_1, fp_1 = recall_and_fp(seg, gt)
            recall_2, fp_2 = recall_and_fp(seg_2, gt)
            #print 'recall_fp %.4f' % (time.time() - t0)

            recalls_1.append(recall_1)
            recalls_2.append(recall_2)
            fps_1.append(fp_1)
            fps_2.append(fp_2)

            if cfg.TEST.SAVE:
                test_set.save(seg, cfg.TEST.SAVE_DIR)



            msg = '%s without blood: (%.4f, %d, %d), with blood: (%.4f, %d, %d)' % (
                   subject, score_1, recall_1, fp_1, score_2, recall_1, fp_2)

            messages.append(msg)
            self.logger.info(msg)

        self.logger.info('mean without blood: (%.4f, %.4f, %.4f), with blood: (%.4f, %.4f, %.4f)' % (
                          np.mean(scores_1), np.mean(recalls_1), np.mean(fps_1),
                          np.mean(scores_2), np.mean(recalls_2), np.mean(fps_2)))




        #if cfg.TEST.SAVE:
        #    with open(os.path.join(cfg.TEST.SAVE_DIR, 'dice.txt'), 'w') as f:
        #        for msg in messages:
        #            f.write(msg + '\n')
        #        f.write('mean: %.4f, var: %.4f' % (np.mean(scores), np.std(scores)))



def recall_and_fp(seg, gt):

    connect_p = measure.label(seg)
    labels = np.unique(connect_p)
    recall, fp = 0, 0
    for ii in range(1, labels.shape[0]):
        region = connect_p == ii
        if region.sum() <= 10:
            continue

        if (region * gt).sum() > 0:
            recall += 1
        else:
            fp += 1

    return recall, fp
