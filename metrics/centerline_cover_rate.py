import numpy as np

import skimage.morphology
from scipy import ndimage
from skimage.morphology import skeletonize_3d, binary_opening, binary_closing, binary_dilation, ball, cube, remove_small_objects
from skimage import measure
from functools import reduce

def skeleton(mask):
    
    volume = skeletonize_3d(mask)
    #volume = remove_small_objects(volume.astype(np.bool), 7, connectivity = 3) 
    points = np.asarray(np.where(volume == True)).T    
    return points

def is_connected(p, q, volume):
    '''
    判断p与q是否相连
    '''
    v_map = np.zeros_like(volume)
    Q = deque()
    Q.append(p)
    v_map[p[0], p[1], p[2]] = 1
    while(Q):
        p = Q.popleft()
        if (p == q).all():
            return True
        
        neibor_box = volume[p[0]-1:p[0]+2, p[1]-1:p[1]+2, p[2]-1:p[2]+2]
        neibor_points = np.asarray(np.where(neibor_box == 1)).T.tolist()
        
        for n_p in neibor_points:
            n_p = [n_p[0]+p[0]-1, n_p[1]+p[1]-1, n_p[2]+p[2]-1]
            if v_map[n_p[0], n_p[1], n_p[2]] == 0:
                Q.append(n_p)
                v_map[n_p[0], n_p[1], n_p[2]] = 1
    return False

def overlapping(P, Q):
    '''
    P, Q: two skeleton points to be compared
    '''
    if len(P) == 0 or len(Q) == 0:
        return 0
    w = max(np.max(P[:,0]), np.max(Q[:,0])) + 10
    h = max(np.max(P[:,1]), np.max(Q[:,1])) + 10
    d = max(np.max(P[:,2]), np.max(Q[:,2])) + 10
    
    vq = np.zeros((w, h, d), 'uint8')
    vq[Q[:,0], Q[:,1], Q[:,2]] = 1
    vp = np.zeros_like(vq)
    vp[P[:,0], P[:,1], P[:,2]] = 1
    
    match = 1
    vv = np.zeros_like(vp)
    for p in P:
        x, y, z = p
        if vq[x-1:x+2, y-1:y+2, z-1:z+2].sum() > 0:
            match = match + 1
        else:
            vv[x, y, z] = 1

    return 1.0 * match / len(P)

def split_vessel(mask):
    #step one: load mask data
    #mask = nib.load('/brain_data/dataset/ccta/mask/%s' % subjects[443]).get_data()
    #mask = mask.astype('bool').astype('uint8')

    #step two: zoom data and erosion
    scale_factor = 0.125
    src_shape = np.array(mask.shape, dtype=float)
    dst_shape = (src_shape*scale_factor).astype('int32').astype('float')
    nmask = ndimage.zoom(mask, np.divide(dst_shape, src_shape), order=0)
    cleaned_nmask = binary_opening(nmask > 0, ball(1))

    #step three: dilation and zoom data
    #cleaned_nmask = binary_closing(cleaned_nmask, ball(1))
    cleaned_nmask = binary_dilation(cleaned_nmask, ball(1))
    cleaned_nmask = binary_dilation(cleaned_nmask, ball(1))
    cleaned_nmask_enlarged = ndimage.zoom(cleaned_nmask, np.divide(src_shape, dst_shape), order=0)
    cleaned_nmask_enlarged = binary_dilation(cleaned_nmask_enlarged, ball(1))

    #step four: get thin vessel mask
    main_vessel_mask = np.logical_and(mask, cleaned_nmask_enlarged > 0).astype('uint8')
    labels, ncomponents = ndimage.measurements.label(main_vessel_mask, np.ones((3, 3, 3), dtype=np.int))
    if labels.max() != 0:
        main_vessel_mask = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    thin_vessel_mask = mask - main_vessel_mask
    thin_vessel_mask[thin_vessel_mask < 0] = 0

    labels = measure.label(thin_vessel_mask)
    props = measure.regionprops(labels)
    for prop in props:
        if prop.bbox[3] - prop.bbox[0] < 8 or prop.bbox[4] - prop.bbox[1] < 8 or prop.bbox[5] - prop.bbox[2] < 16:
            thin_vessel_mask[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2]] = 0

    #step five: use axial to refine main vessel mask
    main_vessel_mask = mask - thin_vessel_mask
    for z in range(mask.shape[2]):
        if main_vessel_mask[:,:,z].sum() == 0:
            continue
        labels = measure.label(main_vessel_mask[:,:,z])
        props = measure.regionprops(labels)   
        prop = reduce(lambda x, y: x if x['area'] > y['area'] else y, props)
        area = prop.area
        for prop in props:
            if 1.0 * prop.area / area < 0.05:
                main_vessel_mask[prop.coords[:,0], prop.coords[:,1], z] = 0

    #step six: get final vessel mask
    thin_vessel_mask = mask - main_vessel_mask
    labels = measure.label(thin_vessel_mask)
    props = measure.regionprops(labels)
    for prop in props:
        if prop.bbox[3] - prop.bbox[0] < 8 or prop.bbox[4] - prop.bbox[1] < 8 or prop.bbox[5] - prop.bbox[2] < 16:
            thin_vessel_mask[prop.coords[:,0], prop.coords[:,1], prop.coords[:,2]] = 0
    main_vessel_mask = mask - thin_vessel_mask
    
    return main_vessel_mask, thin_vessel_mask

def skeleton_overlapping(pred, gt, ccta=True):
    P, Q = skeleton(gt), skeleton(pred)
    
    if ccta:
        _, t0 = split_vessel(gt)
        _, t1 = split_vessel(pred)
  
        P = np.array(list(filter(lambda x: t0[x[0], x[1], x[2]]>0, P)))
        Q = np.array(list(filter(lambda x: t1[x[0], x[1], x[2]]>0, Q)))

    return overlapping(P, Q)

def skeleton_correct_rate(ske, correct_mask):
    correct_num = filter_pts_by_mask(ske, correct_mask).shape[0]
    return correct_num * 1.0 / (ske.shape[0] + 1e-5)

def filter_pts_by_mask(pts, mask):
    return np.array(list(filter(lambda x: mask[x[0], x[1], x[2]] > 0, pts))) #https://blog.csdn.net/weixin_46285081/article/details/104327303

def skeleton_overlapping_v2(pred, gt, top1_pred, top2_pred):  #####
    
    gt_skel, pred_skel = skeleton(gt), skeleton(pred)
    _, gt_thin_vessel = split_vessel(gt)
    _, pred_thin_vessel = split_vessel(pred)
    
    cr = overlapping(filter_pts_by_mask(gt_skel, gt_thin_vessel),
                            filter_pts_by_mask(pred_skel, pred_thin_vessel))
    
    sr = skeleton_correct_rate(gt_skel, pred) #
    sp = skeleton_correct_rate(pred_skel, gt) #
    
    sr1 = skeleton_correct_rate(gt_skel, top1_pred)
    sp1 = skeleton_correct_rate(filter_pts_by_mask(pred_skel, top1_pred),
                                           gt)
    
    sr2 = skeleton_correct_rate(gt_skel, top2_pred)
    sp2 = skeleton_correct_rate(filter_pts_by_mask(pred_skel, top2_pred),
                                           gt)
    
    return sr, sp, sr1, sp1, sr2, sp2, cr #######