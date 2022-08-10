
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

class aucEval:
    
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()
    
    def reset(self):
        self.neg_num = 0
        self.neg_recall_num = 0
        self.pos_num = 0
        self.pos_recall_num = 0
        self.correct_num = 0
        self.total_num = 0
        
    
    def addBatch(self, predict, gt):
        batch_size = predict.size(0)
        predict = predict.cpu().numpy().flatten()
        gt = gt.cpu().numpy().flatten()
        
        label = np.int32(gt)
        correct = (predict == label)
        neg_label = (label == 0)
        neg_num = neg_label.sum()
        neg_recall_num = np.sum(correct * neg_label)
        pos_label = (label > 0)
        pos_num = pos_label.sum()
        pos_recall_num = np.sum(correct * pos_label)
        correct_num = np.sum(correct)
        
        self.neg_num += neg_num
        self.neg_recall_num += neg_recall_num
        self.pos_num += pos_num
        self.pos_recall_num += pos_recall_num
        self.correct_num += correct_num
        self.total_num += batch_size
    
    def getMetric(self):
        epsilon = 0.00000001
        acc = self.correct_num * 100. / (self.total_num + epsilon)
        pos_recall = self.pos_recall_num * 100. / (self.pos_num + epsilon)
        neg_recall = self.neg_recall_num * 100. / (self.neg_num + epsilon)
        
        return acc, pos_recall, neg_recall