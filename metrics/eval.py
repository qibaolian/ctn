import numpy as np
from skimage.morphology import skeletonize_3d, remove_small_objects
from skimage.measure import regionprops, label
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def eval_result_from_volume(pred, gt, thres=10):
    
    connect_p = label(pred)
    labels =  np.unique(connect_p)
    # connect_p, labels = label(pred, return_num=True)
    print (len(labels))
    n_predict, n_precision = 0, 0
    for ii in range(1, labels.shape[0]):
        if (connect_p == ii).sum() >= thres:
            n_predict += 1
            if ((connect_p == ii) * gt).sum() > 0:
                n_precision += 1
    
    connect_gt = label(gt)
    labels = np.unique(connect_gt)
    # connect_gt, labels = label(gt, return_num=True)
    print (len(labels))
    
    n_gt = labels.shape[0] - 1
    n_recall = 0
    for ii in range(1, labels.shape[0]):
        if ((connect_gt == ii) * pred).sum() >= thres:
            n_recall += 1
    
    dice = -1
    if gt.sum() > 0:
        overlap = (pred * gt).sum()
        dice = 2.0 * overlap / (pred.sum() + gt.sum())
        
    
    return dice,  n_recall, n_gt, n_precision, n_predict

def eval_result_from_patch(pred, gt, thres=10):
    
    n_recall, n_gt, n_precision, n_predict, n_correct, n_total = 0, 0, 0, 0, 0, 0
    dice = 0
    for (p, g) in zip(pred, gt):
        
        if g.sum() > 0:
            overlap = (p * g).sum()
            dice += 2.0 *  overlap / (p.sum() + g.sum())
            n_gt += 1
            if overlap >= thres:
                n_recall += 1
                n_precision += 1
                n_correct += 1
        elif p.sum() == 0:
            n_correct += 1
        
        if p.sum() > 0:
            n_predict += 1
        
        n_total += 1
    
    return dice, n_recall, n_gt, n_precision, n_predict, n_correct, n_total
    
def eval_results(p_list, volume=True):
    
    with ProcessPoolExecutor() as ex:
        results = ex.map(eval_result_from_volume if volume else eval_result_from_patch, 
                               [p[0] for p in p_list], [p[1] for p in p_list])
    
    return np.array(list(results), dtype='float32')
