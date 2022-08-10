import numpy as np
from skimage.morphology import skeletonize_3d, remove_small_objects
from skimage import measure
from skimage.measure import regionprops, label
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .hd_distance import hd, hd95, assd
from .centerline_cover_rate import skeleton_overlapping, skeleton_overlapping_v2

def dice(seg, gt):

    union = 1.0 * (seg*gt).sum()
    num = seg.sum() + gt.sum()
    
    return 2 * union / num

def dice_v2(seg, gt):
    seg_mask, gt_mask = seg > 0, gt > 0
    inter = (seg_mask * gt_mask).sum()
    union = seg_mask.sum() + gt_mask.sum()
    dice_coeff = 2. * inter / union
    
    seg2_mask, gt2_mask = seg > 1, gt > 1
    r2 = np.sum(gt2_mask * seg_mask) * 1. / gt2_mask.sum()
    p2 = np.sum(seg2_mask * gt_mask) * 1. / seg2_mask.sum()
    
    return [dice_coeff, r2, p2]

def post_process(seg):
    labels = measure.label(seg>0)
    props = measure.regionprops(labels)
    for prop in props:
        if prop.bbox[3] - prop.bbox[0] < 16:
            seg[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2]] = 0
    return seg

def post_process_v2(seg):
    labels = measure.label(seg>0)
    props = measure.regionprops(labels)
    area_labels = []
    for prop in props:
        area_labels.append([prop.area, prop.label])
        if prop.bbox[3] - prop.bbox[0] < 16:
            seg[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2]] = 0
    area_labels.sort()
    top1_area_mask = (labels == area_labels[-1][1])
    if len(area_labels) <= 1:
        top2_area_mask = top1_area_mask
    else:
        top2_area_mask = top1_area_mask + (labels == area_labels[-2][1])
    return seg, top1_area_mask, top2_area_mask

'''
def eval_volume(seg, gt, use_post_process=True, use_hd=True):
    
    if use_post_process:
        seg = post_process(seg)
    
    dd_vessel = dice(seg==1, gt==1)
    #dd_aorta = dice(seg==2, gt==2)
    
    if use_hd:
        if (seg == 1).sum() == 0:
            return dd_vessel, 500, 50
        return dd_vessel, hd(seg==1, gt==1), assd(seg==1, gt==1)#, dd_aorta
    else:
        return dd_vessel#, dd_aorta
'''

def eval_ccta_volume_v2(seg, gt, num_class, use_hd=True):
    seg, top1_pred, top2_pred = post_process_v2(seg)
    results = []
    for i in range(1, num_class):
        results.append(dice(seg == i, gt == i))
    vdice, r2, p2 = dice_v2(seg, gt)
    results.append(vdice)
    
    if use_hd:
        if (seg > 0).sum() == 0:
            results.extend([100]*num_class)
        else:
            for i in range(1, num_class):
                results.append(assd(seg == i, gt == i))
            if num_class == 2:
                results.append(results[-1])
            else:
                results.append(assd(seg > 0, gt > 0))
    
    sr, sp, sr1, sp1, sr2, sp2, cr = skeleton_overlapping_v2(
        (seg>0).astype('uint8'),
        (gt>0).astype('uint8'),
        top1_pred,
        top2_pred
    )
    
    results += [sr, sp, sr1, sp1, sr2, sp2, r2, p2, cr]
    
    return results

def eval_ccta_volume(seg, gt, num_class, use_post_process=True, use_hd=True):
    if use_post_process:
        seg =  post_process(seg)
    
    results = []
    for i in range(1, num_class):
        results.append(dice(seg == i, gt == i))
    results.append(dice(seg > 0, gt > 0))
    
    if use_hd:
        if (seg > 0).sum() == 0:
            results.extend([100]*num_class)
        else:
            for i in range(1, num_class):
                results.append(assd(seg == i, gt == i))
            if num_class == 2:
                results.append(results[-1])
            else:
                results.append(assd(seg > 0, gt > 0))
            
    cr = skeleton_overlapping((seg>0).astype('uint8'), (gt>0).astype('uint8'))
    results.append(cr)
    
    return results
    
def eval_cta_volume(seg, gt, num_class, use_post_process=True, use_hd=True):
    if use_post_process:
        seg =  post_process(seg)
    
    results = []
    for i in range(1, num_class):
        results.append(dice(seg == i, gt == i))
    results.append(dice(seg > 0, gt > 0))
    
    if use_hd:
        if (seg > 0).sum() == 0:
            results.append(500)
            results.append(50)
        else:
            results.append(hd(seg > 0, gt > 0))
            results.append(assd(seg > 0, gt > 0))
    
    cr = skeleton_overlapping((seg>0).astype('uint8'), (gt>0).astype('uint8'), False)
    results.append(cr)
    
    return results

def eval_volume(seg, gt, num_class, use_post_process=True, use_hd=True, use_cl=True):
    if use_post_process:
        seg =  post_process(seg)  #######
    
    results = []
    for i in range(1, num_class):
        results.append(dice(seg == i, gt == i))
    results.append(dice(seg > 0, gt > 0))
    
    if use_hd:
        if (seg > 0).sum() == 0:
            results.append(500)
            results.append(50)
        else:
            results.append(hd(seg > 0, gt > 0))
            results.append(assd(seg > 0, gt > 0))

    if use_cl:
        cr = skeleton_overlapping((seg>0).astype('uint8'), (gt>0).astype('uint8'), False)
        results.append(cr)
    
    return results

def eval_xinji(seg, gt):
    
    return dice(seg==1, gt==1), dice(seg==2, gt==2), dice(seg==3, gt==3)