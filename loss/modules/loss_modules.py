from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.config import cfg
import numpy as np

from .lovasz_losses import lovasz_softmax
from scipy.ndimage import distance_transform_edt

class FCCELoss(nn.Module):
    
    def __init__(self):
        super(FCCELoss, self).__init__()
        
        weight = None
        if 'CLASS_WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.CLASS_WEIGHT) > 0:
            weight = cfg.LOSS.CLASS_WEIGHT
            weight = torch.FloatTensor(weight).cuda()
        
        ignore_index = -100
        if 'IGNORE_INDEX' in cfg.LOSS.keys():
            ignore_index = cfg.LOSS.IGNORE_INDEX
        
        reduction = 'elementwise_mean'
        if 'REDUCTION' in cfg.LOSS.keys():
            reduction = cfg.LOSS.REDUCTION
        
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)#, reduction=reduction)
   
    def forward(self, input, target):
        return self.ce_loss(input, target)

class WCELoss(nn.Module):
    def __init__(self, per_image=True):
        super(WCELoss, self).__init__()
        
        self.weight = None
        if 'CLASS_WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.CLASS_WEIGHT) > 0:
            self.weight = cfg.LOSS.CLASS_WEIGHT
            self.weight = torch.FloatTensor(self.weight).cuda()
        
        self.per_image = per_image
        
    def forward(self, input, target):
        
        b, c = input.size()[:2]
        
        total_loss = 0
        if self.per_image:
            for i in range(b):
                loss = F.cross_entropy(input[i:i+1], target[i:i+1], weight=self.weight, reduce=False, size_average=False)
                fg = target[i:i+1] > 0
                bg = 1 - fg
                fg_num, bg_num = fg.sum().item(), bg.sum().item()
                fg_loss, bg_loss = loss[fg], loss[bg]
                
                k = min(bg_num, max(fg_num, 1024))
                
                if fg_num > 0:
                    total_loss += fg_loss.sum() / fg_num
                
                if bg_num > 0:
                    top_k, _ = bg_loss.topk(k)
                    total_loss += top_k.sum() / k
            total_loss = total_loss / b
        else:
            loss = F.cross_entropy(input, target, weight=self.weight, reduce=False, size_average=False)
            fg = target > 0
            bg = 1 - fg
            fg_num, bg_num = fg.sum().item(), bg.sum().item()
            fg_loss, bg_loss = loss[fg], loss[bg]
            
            k = min(bg_num, max(fg_num, 1024*b))
            if  fg_num > 0:
                total_loss += fg_loss.sum() / fg_num
            
            if bg_num > 0:
                top_k, _ = bg_loss.topk(k)
                total_loss += top_k.sum() / k
    
        return total_loss

class Bootstrapped_FCCELoss(nn.Module):
    
    def __init__(self, k=1024):
        super(Bootstrapped_FCCELoss, self).__init__()
        
        self.weight = None
        if 'CLASS_WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.CLASS_WEIGHT) > 0:
            self.weight = cfg.LOSS.CLASS_WEIGHT
            self.weight = torch.FloatTensor(self.weight).cuda()
        
        self.k = k
   
    def forward(self, input, target):
        b, c = input.size(0), input.size(1)
        if input.dim() > 2:
            input = input.view(b, c, -1)
            input = input.transpose(1,2)
            target = target.view(b, -1)
        
        total_loss = 0
        for i in range(b):
            loss = F.cross_entropy(input[i], target[i], 
                                   weight=self.weight, reduce=False, size_average=False)
            topk_loss, _ = loss.topk(self.k)
            total_loss += topk_loss.sum() / self.k
        
        return total_loss / b
        
    
class FocalLoss(nn.Module):
    
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        
        if 'CLASS_WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.CLASS_WEIGHT) > 0:
            alpha = cfg.LOSS.CLASS_WEIGHT
        
        if isinstance(alpha,(float,int,)): self.alpha = torch.FloatTensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.FloatTensor(alpha)
        self.size_average = size_average
        
    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        #pt = Variable(logpt.data.exp())
        pt = logpt.exp()
        
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)
        
        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def compute_dice(input, target):
    dice = torch.zeros([input.size(0), input.size(1)]).to(torch.device("cuda"))
    pred = F.softmax(input, dim=1)
    for i in range(1, input.size(1)):
        input_i = pred[:,i,...].contiguous().view(input.size(0), -1)
        target_i = (target == i).float().view(input.size(0), -1)

        num = (input_i * target_i)
        num = torch.sum(num, dim=1)

        den1 = input_i * input_i
        den1 = torch.sum(den1, dim=1)

        den2 = target_i * target_i
        den2 = torch.sum(den2, dim=1)

        epsilon = 1e-6
        dice[:, i] = (2 * num + epsilon) / (den1 + den2 + epsilon)

    return dice
        
class Bootstrapped_BinaryDiceLoss(nn.Module):
    
    def __init__(self, k=16):
        super(Bootstrapped_BinaryDiceLoss, self).__init__()
        self.k = k
    
    def forward(self, input, target):
        
        pred = F.softmax(input, dim=1)
        pred = pred.view(pred.size(0), pred.size(1), pred.size(2), -1) # B,C,D,H,W => B,C,D,H*W
        target = target.view(target.size(0), target.size(1), -1) # B,D,H,W => B,D,H*W
        
        b, d = pred.size(0), pred.size(2)
        total_loss = 0
        for i in range(b):
            
            pred_i = pred[i, 1, ...].contiguous().view(d, -1)
            gt_i = (target[i, ...] == 1).float().view(d, -1)
            
            num = pred_i * gt_i
            num = torch.sum(num, dim=1)
            den1 = pred_i * pred_i
            den1 = torch.sum(den1, dim=1)
            den2 = gt_i * gt_i
            den2 = torch.sum(den2, dim=1)
            
            dice = (2*num + 1e-6) / (den1 + den2 + 1e-6)
            loss = 1.0 - dice
            
            topk_loss, _ = loss.topk(self.k)
            total_loss += topk_loss.sum() / self.k
        
        return total_loss / b

class BinaryDiceLoss(nn.Module):
    """ Dice Loss of binary class
    """
    
    def __init__(self, smooth=1, p=2, reduction='mean', per_image=True):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.per_image = per_image
    
    def compute_dice(self, probs, labels):
        
        probs = probs.contiguous().view(-1)
        labels = labels.contiguous().view(-1).float()
        
        num = 2 * torch.sum(torch.mul(probs, labels)) + self.smooth
        den = torch.sum(probs.pow(self.p) + labels.pow(self.p)) + self.smooth
        
        loss = 1.0 - num/den
        
        return loss

    def forward(self, predict, target):
        
        if self.per_image:
            loss, num = 0, 0
            for prob, lab in zip(predict, target):
                if lab.sum() == 0:
                    continue
                loss += self.compute_dice(prob.unsqueeze(0), lab.unsqueeze(0))
                num += 1
            num = max(num, 1)
            return loss / num if self.reduction=='mean' else loss
        else:
            return self.compute_dice(predict, target)


class BinaryGateDiceLoss(nn.Module):
    """ Dice Loss of binary class
    """

    def __init__(self, smooth=1, p=2, reduction='mean', per_image=True):
        super(BinaryGateDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.per_image = per_image

    def compute_dice(self, probs, labels, gate):

        probs = probs.contiguous().view(-1)
        labels = labels.contiguous().view(-1).float()
        gate = gate.contiguous().view(-1).float()

        num = 2 * torch.sum(gate*torch.mul(probs, labels)) + self.smooth
        den = torch.sum(gate*(probs.pow(self.p) + labels.pow(self.p))) + self.smooth

        loss = 1.0 - num / den

        return loss

    def forward(self, predict, target, gate):

        if self.per_image:
            loss, num = 0, 0
            for prob, lab, g in zip(predict, target, gate):
                if lab.sum() == 0:
                    continue
                loss += self.compute_dice(prob.unsqueeze(0), lab.unsqueeze(0), g.unsqueeze(0))
                num += 1
            num = max(num, 1)
            return loss / num if self.reduction == 'mean' else loss
        else:
            return self.compute_dice(predict, target, gate)

class BCELoss(nn.Module):
    """Weight Binary Cross Entropy
    """
    
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        
        weight = 1.0
        if 'CLASS_WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.CLASS_WEIGHT) > 0:
            weight = cfg.LOSS.CLASS_WEIGHT[-1]
        self.weight = float(weight)
        self.reduction = reduction
     
    def forward(self, output, target):
        output = output.view(output.shape[0], -1)
        target = target.view(target.shape[0], -1).float()
        
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        #target = torch.clamp(target, min=eps, max=1.0 - eps)    
        loss = -self.weight * target.mul(torch.log(output)) - ((1.0 - target).mul(torch.log(1.0 - output)))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def compute_edts_forhdloss(segmentation):
    res = np.zeros(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = distance_transform_edt(posmask) + distance_transform_edt(negmask)
    return res

class BHDLoss(nn.Module):
    """compute haudorff loss for binary segmentation
    https://arxiv.org/pdf/1904.10030v1.pdf
    """
    
    def __init__(self):
        super(BHDLoss, self).__init__()
    
    def forward(self, output, target):
        
        #output = torch.sigmoid(output)
        
        pc = output[:,0,...]
        gt = target.float()
        with torch.no_grad():
            pc_dist = compute_edts_forhdloss(pc.cpu().numpy()>0.5)
            gt_dist = compute_edts_forhdloss(gt.cpu().numpy()>0.5)
        
        pred_error = (gt - pc) ** 2
        dist = pc_dist**2 + gt_dist**2
        
        dist = torch.from_numpy(dist)
        if dist.device != pred_error.device:
            dist = dist.to(pred_error.device).float()
        
        multipled = torch.einsum("bxyz,bxyz->bxyz", pred_error, dist)
        hd_loss = multipled.mean()
        
        return hd_loss

class HDLoss(nn.Module):
    
    def __init__(self):
        super(HDLoss, self).__init__()
        
    def forward(self, input, target):
        pred = F.softmax(input, dim=1)
        
        loss = 0
        bhd = BHDLoss()
        for i in range(1, input.size(1)):
            pred_i = pred[:, i:i+1, ...]
            target_i = (target == i).float()
            loss += bhd(pred_i, target_i)
        return loss

class BCLDiceLoss(nn.Module):
    
    def __init__(self, max_iter=10):
        super(BCLDiceLoss, self).__init__()
        self.max_iter = max_iter
    
    def soft_erode_2d(self, img):
        p1 = -F.max_pool2d(-img , ( 3 , 1 ) , ( 1 , 1 ) , ( 1 , 0 ) )
        p2 = -F.max_pool2d(-img , ( 1 , 3 ) , ( 1 , 1 ) , ( 0 , 1 ) )
        return torch.min( p1 , p2 )
    
    def soft_dilate_2d(self, img):
        return F.max_pool2d(img , ( 3 , 3 ) , ( 1 , 1 ) , ( 1 , 1 ) )
    
    def soft_open_2d(self, img):
        return self.soft_dilate_2d(self.soft_erode_2d(img))
    
    def soft_skel_2d(self, img, max_iter):
        img1 = self.soft_open_2d(img)
        skel = F.relu(img-img1)
        for j in range(max_iter):
            img = self.soft_erode_2d(img)
            img1 = self.soft_open_2d(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel*delta)
        
        return skel

    def soft_erode_3d(self, img):
        p1 = -F.max_pool3d(-img, ( 1, 3, 3 ) , ( 1, 1, 1 ) , ( 0, 1, 1 ) )
        p2 = -F.max_pool3d(-img, ( 3, 1, 3 ) , ( 1, 1, 1 ) , ( 1, 0, 1 ) )
        p3 = -F.max_pool3d(-img, ( 3, 3, 1 ) , ( 1, 1, 1 ) , ( 1, 1, 0 ) )
        return torch.min(torch.min( p1 , p2 ), p3)
    
    def soft_dilate_3d(self, img):
        return F.max_pool3d(img , ( 3 , 3 , 3 ) , ( 1 , 1 , 1 ) , ( 1 , 1  , 1 ) )
    
    def soft_open_3d(self, img):
        return self.soft_dilate_3d(self.soft_erode_3d(img))
    
    def soft_skel_3d(self, img, max_iter):
        img1 = self.soft_open_3d(img)
        skel = F.relu(img-img1)
        for j in range(max_iter):
            img = self.soft_erode_3d(img)
            img1 = self.soft_open_3d(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel*delta)
        
        return skel
    
    def soft_skel(self, img, max_iter):
        if img.dim() == 4:
            return self.soft_skel_2d(img, max_iter)
        elif img.dim() == 5:
            return self.soft_skel_3d(img, max_iter)
        return 0

    def soft_clDice(self, v_p, v_l, max_iter=10, smooth=1):
        s_p = self.soft_skel(v_p, max_iter)
        s_l = self.soft_skel(v_l, max_iter)
        tprec = ((s_p * v_l).sum() + smooth) / (s_p.sum() + smooth)
        tsens = ((s_l * v_p).sum() + smooth) / (s_l.sum() + smooth)
        return 2 * tprec*tsens / (tprec + tsens)
    
    def forward(self, output, target):
        pc = output[:,0,...]
        gt = target[:, None, ...]
        
        return self.soft_clDice(pc, gt, self.max_iter)



        
class DiceLoss(nn.Module):
   
    def __init__(self):
        super(DiceLoss, self).__init__()
        
        weight = None
        if 'DICE_WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.DICE_WEIGHT) > 0:
            weight = cfg.LOSS.DICE_WEIGHT
        self.weight = weight
    
    def forward(self, input, target):
        
        #dice = compute_dice(input, target)
        #dice = dice[:, 1:] #we ignore bg dice val, and take the fg
        #dice = torch.sum(dice, dim=1)
        #dice = dice / (input.size(1) - 1)
        #dice_total = -1.0 * torch.sum(dice) / dice.size(0) #divide by batch_sz
        #return 1.0 + dice_total
        
        dice = BinaryDiceLoss(per_image=True)
        total_loss = 0
        pred = F.softmax(input, dim=1)
        
        for i in range(1, pred.size(1)):
            dice_loss = dice(pred[:,i], target==i)
            if self.weight is not None:
                dice_loss *= self.weight[i]
            total_loss += dice_loss
        return total_loss / (pred.size(1) - 1)

class ELDiceLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(ELDiceLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, input, target):
        smooth = 1.0
        
        pred = F.softmax(input, dim=1)
        loss = 0 #torch.Tensor([0]).float().to(torch.device("cuda"))
        for i in range(1, pred.size(1)):
            pred_i = pred[:,i,:,:,:]
            target_i = (target == i).float()
            
            intersect = (pred_i * target_i).sum()
            union = torch.sum(pred_i) + torch.sum(target_i)
            dice = (2*intersect + smooth) / (union + smooth)
            
            if target_i.sum().item() != 0:
                loss += (-torch.log(dice))**self.gamma
            else:
                loss += 1 - dice
        loss = loss / (input.size(1) - 1)
        return loss
        
    
class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=0.5, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
    
    def forward(self, input, target):
        dice = compute_dice(input, target)
        dice = dice[:, 1:]
        
        pt = dice.contiguous().view(target.size(0), -1)
        #assert (pt < 1).any() and (pt > 1e-12).any(), pt
        logpt = torch.log(pt)
        
        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class GateDiceLoss(nn.Module):

    def __init__(self):
        super(GateDiceLoss, self).__init__()

        weight = None
        if 'DICE_WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.DICE_WEIGHT) > 0:
            weight = cfg.LOSS.DICE_WEIGHT
        self.weight = weight

    def forward(self, input, target, gate):

        # dice = compute_dice(input, target)
        # dice = dice[:, 1:] #we ignore bg dice val, and take the fg
        # dice = torch.sum(dice, dim=1)
        # dice = dice / (input.size(1) - 1)
        # dice_total = -1.0 * torch.sum(dice) / dice.size(0) #divide by batch_sz
        # return 1.0 + dice_total

        dice = BinaryGateDiceLoss(per_image=True)
        total_loss = 0
        pred = F.softmax(input, dim=1)

        for i in range(1, pred.size(1)):
            dice_loss = dice(pred[:, i], target == i, gate)
            if self.weight is not None:
                dice_loss *= self.weight[i]
            total_loss += dice_loss
        return total_loss / (pred.size(1) - 1)

class SurfaceLoss(nn.Module):
    
    def __init__(self):
        super(SurfaceLoss, self).__init__()
    
    def forward(self, input, dist_map):
        
        loss = input * dist_map
        
        return loss.mean()

class SDMLoss(nn.Module):
    
    def __init__(self):
        super(SDMLoss, self).__init__()
    
    def forward(self, input, target):
        smooth = 1e-5
        intersect = torch.sum(input * target)
        pd_sum = torch.sum(input ** 2)
        gt_sum = torch.sum(target ** 2)
        dd = (intersect + smooth) / (intersect + pd_sum + gt_sum + smooth)
        loss = -dd + torch.norm(input-target, 1)/torch.numel(input)
        
        return loss
    
class LovaszLoss(nn.Module):
    
    def __init__(self):
        super(LovaszLoss, self).__init__()
    
    def forward(self, input, target):
        pred = F.softmax(input, dim=1)
        
        if input.dim() >= 5: # B, C, D, H, W
            pred = pred.view(pred.size(0),pred.size(1),pred.size(2), -1)  # B,C,D,H,W => B,C,D,H*W
            target = target.view(target.size(0), target.size(1), -1) # B,D,H,W => B,D,H*W
        
        return lovasz_softmax(pred, target, per_image=True)
        
class MixLoss(nn.Module):
    
    def __init__(self, loss):
        
        super(MixLoss, self).__init__()
        self.loss = loss
        
    def forward(self, input, target, gate=None):
        
        if 'WEIGHT' in cfg.LOSS.keys() and len(cfg.LOSS.WEIGHT) > 0:
            weight = cfg.LOSS.WEIGHT
        else:
            weight = [1.0] * len(self.loss)
        
        loss = 0
        for i in range(len(self.loss)):
            if gate is None:
                loss += weight[i] * self.loss[i](input, target)
            else:
                loss += weight[i] * self.loss[i](input, target, gate)

        return loss
            
        
        
