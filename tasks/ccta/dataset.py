from utils.config import cfg
from tasks.aneurysm.datasets.base_dataset import BaseDataset
from tasks.aneurysm.datasets.data_utils import *
import random
import numpy as np
import torch
import nibabel as nib
import os
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import scipy.io as scio
from utils.tools.util import progress_monitor
from torch.utils import data
import torch.nn.functional as F
import time
from tasks.aneurysm.rotation_3d import getTransform, transform_point, transform_image

def load_subject_validate(subject):
    image = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER,
                                          '%s.nii.gz' % subject)).get_data()
    mask = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER,
                                          '%s.nii.gz' % subject)).get_data()
    image = np.transpose(image, (2, 0, 1))  
    mask = np.transpose(mask, (2, 0, 1))
    
    image, mask = image[::-1, ...], mask[::-1, ...]
    mask[mask > 0] = 1
    
    z_start = 0
    while z_start < image.shape[0]:
        if mask[z_start, :, :].sum() > 0:
            break
        else:
            z_start += 1
    
    z_end = image.shape[0] - 1
    while z_end > 0:
        if mask[z_end, :, :].sum() > 0:
            break
        else:
            z_end -= 1
    
    z_start = max(0, z_start-32)
    z_end = min(z_end + 32, image.shape[0]-1)
    
    data_lst = []
    step = cfg.TRAIN.DATA.PATCH_SIZE[0] // 2
    z_coords = list(range(z_start+step, z_end-step, step))
    
    for z in z_coords:
        img = set_window_wl_ww(image[z-step:z+step, :, :].copy(), cfg.TRAIN.DATA.WL_WW[0], cfg.TRAIN.DATA.WL_WW[1])
        data_lst.append((img, mask[z-step:z+step, :, :].copy()))
    
    return data_lst

def FindConjunctionPoints(points, volume):
    '''
        在中心线上寻找交叉点
    '''
    
    pts = list(filter(lambda x: np.sum(volume[x[0]-1:x[0]+2, 
                                         x[1]-1:x[1]+2,
                                         x[2]-1:x[2]+2])>=4, points))
    pts = filter(lambda x: np.sum(volume[x[0]-1:x[0]+2, 
                                         x[1]-1:x[1]+2,
                                         x[2]-1:x[2]+2])<=6, pts)
    return np.array(list(pts))

def tensor_resize(image, shape, is_label=True):
    x = torch.from_numpy(image[np.newaxis, np.newaxis, ...])
    if is_label:
        y = F.interpolate(x.float(), shape, mode='nearest')
    else:
        y = F.interpolate(x.float(), shape, mode= 'trilinear', align_corners=True)
    y = y.numpy().astype(image.dtype)[0, 0]
    return y

class VesselPatchSampler(object):
    
    def __init__(self, subject, sample_num, stage='train'):
        
        img_nii = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '%s.nii.gz' % subject))
        image = img_nii.get_data()
        mask = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER, '%s.nii.gz' % subject)).get_data()
        points = scio.loadmat(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '../skeleton/%s.mat' % subject))['points']
        vessel = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '../mask_all3/%s.nii.gz' % subject)).get_data()
        vessel = (vessel == 2).astype('uint8')
        
        self.image = np.transpose(image, (2, 0, 1)) 
        self.mask = np.transpose(mask, (2, 0, 1))
        self.vessel = np.transpose(vessel, (2, 0, 1))
        points = np.roll(points, 1, axis=1)
        self.points = points
        #self.points = list(filter(lambda x: self.vessel[x[0], x[1], x[2]] == 1, points)) #thin vessel skeleton points
        
        if img_nii.header['pixdim'][0] > 0:
            self.image = self.image[::-1, ...]
            self.mask = self.mask[::-1, ...]
            self.vessel = self.vessel[::-1, ...]
            self.points = list(map(lambda x: np.array([self.mask.shape[0]-1-x[0],x[1],x[2]], 'int32'), self.points))
        
        if stage == 'train':
            self.coords = self.sample_coords(sample_num)
            self.indices = list(range(sample_num))
            random.shuffle(self.indices)
            self.num = sample_num
        else:
            self.coords = self.sample_volume()
            self.num = len(self.coords)
            
        
        self.sample_idx = 0
        self.wl, self.ww = cfg.TRAIN.DATA.WL_WW
    
    
    def sample_volume(self):
        
        P = np.array(self.points)
        volume = np.zeros_like(self.vessel)
        volume[P[:,0], P[:, 1], P[:,2]] = 1

        vx, vy, vz = self.vessel.shape
        px, py, pz = 8, 8, 8
        coords = []
        for x in range(0, vx, px):
            for y in range(0, vy, py):
                for z in range(0, vz, pz):
                    v = volume[x:x+px, y:y+py, z:z+pz]
                    if v.sum() == 0:
                        continue
                    cx, cy, cz = np.where(v == 1)
                    idx = len(cx) // 2
                    coords.append(((x+cx[idx], y+cy[idx], z+cz[idx]), 1))
        
        return coords
    
    def __len__(self):
        return self.num
    
    def volume_size(self):
        return self.vessel.shape
    
    def volume_mask(self):
        return self.vessel
    
    def sample_coords(self, num):
        
        coords = []
        
        pp = np.array(self.points)
        vv = np.zeros_like(self.vessel)
        vv[pp[:,0], pp[:,1], pp[:,2]] = 1
        cp = FindConjunctionPoints(pp, vv)        
        k1 = min(num // 3, len(cp))
        k2 = num - k1
        if k1 > 0:
            samples = np.random.choice(range(len(cp)), size=k1, replace = k1 > len(cp))
            coords.extend([(cp[i], 1) for i in samples])
        samples = np.random.choice(range(len(self.points)), size=k2, replace = k2 > len(self.points))
        coords.extend([(self.points[i], 1) for i in samples])
        
        #positive samples
        #samples = np.random.choice(range(len(self.points)), size=num, replace = num > len(self.points))
        #coords.extend([(self.points[i], 1) for i in samples])
        
        return coords
    
    def sample_validate(self):
        
        coord, _ = self.coords[self.sample_idx]
        self.sample_idx = (self.sample_idx + 1) % self.num
        
        vx, vy, vz = self.vessel.shape
        px, py, pz = cfg.TEST.DATA.PATCH_SIZE
        
        x, y, z = coord
        sx, ex = max(0, x - px // 2), min(vx, x + px - px // 2)
        sy, ey = max(0, y - py // 2), min(vy, y + py - py // 2)
        sz, ez = max(0, z - pz // 2), min(vz, z + pz - pz // 2)
        
        img = self.image[sx:ex, sy:ey, sz:ez]
        msk = self.vessel[sx:ex, sy:ey, sz:ez]
        if img.shape != (px, py, pz):
            return self.sample_validate()
        
        img = set_window_wl_ww(img, self.wl, self.ww)
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')[np.newaxis, ...]
        msk = msk.astype('bool').astype('uint8')
        
        return {'img': img, 'gt': msk, 'coord': np.array([[sx, ex],[sy, ey],[sz, ez]], 'int32')}
    
    def sample_train(self):
        
        coord, _ = self.coords[self.sample_idx] 
        self.sample_idx = (self.sample_idx + 1) % self.num
        
        vx, vy, vz = self.vessel.shape
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE
        
        x, y, z = coord
        x = x + (np.random.choice(2) * 2 - 1) * np.random.choice(3)
        y = y + (np.random.choice(2) * 2 - 1) * np.random.choice(3)
        z = z + (np.random.choice(2) * 2 - 1) * np.random.choice(3)        
        x = max(0, min(x, vx-1))
        y = max(0, min(y, vy-1))
        z = max(0, min(z, vz-1))
        
        sx, ex = max(0, x - px // 2), min(vx, x + px - px // 2)
        sy, ey = max(0, y - py // 2), min(vy, y + py - py // 2)
        sz, ez = max(0, z - pz // 2), min(vz, z + pz - pz // 2)

        img = self.image[sx:ex, sy:ey, sz:ez]
        msk = self.vessel[sx:ex, sy:ey, sz:ez]
        
        lpx = max(0, px // 2 - x)
        rpx = px - img.shape[0] - lpx
        lpy = max(0, py // 2 - y)
        rpy = py - img.shape[1] - lpy
        lpz = max(0, pz // 2 - z)
        rpz = pz - img.shape[2] - lpz
        
        img = np.pad(img, ((lpx, rpx), (lpy, rpy), (lpz, rpz)), 'constant', constant_values=0)
        msk = np.pad(msk, ((lpx, rpx), (lpy, rpy), (lpz, rpz)), 'constant', constant_values=0)
        
        img = set_window_wl_ww(img, self.wl, self.ww)
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')[np.newaxis, ...]
        msk = msk.astype('bool').astype('uint8')
        
        flip_y = np.random.choice(2)*2 - 1
        flip_z = np.random.choice(2)*2 - 1
        img = img[..., ::flip_y, ::flip_z].copy()
        msk = msk[..., ::flip_y, ::flip_z].copy()
        
        return {'img': img, 'gt': msk}
        
    def sample_positive(self, k):
        if len(self.skeleton) > 0:
            samples = np.random.choice(range(len(self.skeleton)), size=k, replace= k > len(self.skeleton))
            coords = [self.skeleton[i] for i in samples]
        else:
            print ('skeleton not found')
            aa = np.where(self.mask > 0)
            samples = np.random.choice(range(len(aa)), size=k, replace= k > len(aa))
            coords = [(aa[i][0], aa[i][1], aa[i][2]) for i in samples]
        
        vx, vy, vz = self.mask.shape
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE
        data_lst = []
        for coord in coords:
            x, y, z = coord
            y = y + (np.random.choice(2) * 2 - 1)*np.random.choice(5)
            z = z + (np.random.choice(2) * 2 - 1)*np.random.choice(5)
            
            x = min( max(0, x - px // 2),  vx - px)
            y = min( max(0, y - py // 2),  vy - py)
            z = min( max(0, z - pz // 2),  vz - pz)
            
            img = self.image[x:x+px, y:y+py, z:z+pz]
            img = set_window_wl_ww(img, self.wl, self.ww)
            gt = self.mask[x:x+px, y:y+py, z:z+pz].copy()
            gt[gt > 0] = 1
            
            data_lst.append((img, gt))
        
        return data_lst
  
    def sample_negative(self, k):
        aa = np.where(self.mask == 0)
        samples = np.random.choice(range(len(aa)), size=k, replace= k > len(aa))
        coords = [(aa[i][0], aa[i][1], aa[i][2]) for i in samples]
        
        vx, vy, vz = self.mask.shape
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE
        data_lst = []
        for coord in coords:
            x, y, z = coord
            x = min( max(0, x - px // 2),  vx - px)
            y = min( max(0, y - py // 2),  vy - py)
            z = min( max(0, z - pz // 2),  vz - pz)
            
            img = self.image[x:x+px, y:y+py, z:z+pz]
            img = set_window_wl_ww(img, self.wl, self.ww)
            gt = self.mask[x:x+px, y:y+py, z:z+pz].copy()
            gt[gt > 0] = 1
            
            data_lst.append((img, gt))
            
        return data_lst
    
    def sample_traverse(self, k):
        
        aa = get_patch_coords(cfg.TRAIN.DATA.PATCH_SIZE, self.mask.shape)
        samples = np.random.choice(range(len(aa)), size=k, replace= k > len(aa))
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE
        data_lst = []
        for sample in samples:
            x, y, z = aa[sample]
            img = self.image[x:x+px, y:y+py, z:z+pz]
            img = set_window_wl_ww(img, self.wl, self.ww)
            gt = self.mask[x:x+px, y:y+py, z:z+pz].copy()
            gt[gt > 0] = 1
            
            data_lst.append((img, gt))
        return data_lst
    
    def sample_random(self, k):
        vx, vy, vz = self.image.shape
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE
        data_lst = []
        for i in range(k):
            x = random.randint(0, vx - px)
            y = random.randint(0, vy - py)
            z = random.randint(0, vz - pz)

            img = self.image[x:x+px, y:y+py, z:z+pz]
            img = set_window_wl_ww(img, self.wl, self.ww)
            gt =  self.mask[x:x+px, y:y+py, z:z+pz].copy()
            gt[gt > 0] = 1
            data_lst.append((img, gt))
        return data_lst
            
    def sample(self, k):
        
        data_lst = []
        #data_lst.extend(self.sample_positive(k))
        #data_lst.extend(self.sample_negative(k))
        #data_lst.extend(self.sample_traverse(k))
        data_lst.extend(self.sample_random(k))
        
        return data_lst

def get_nii_path(base_dir, subject):
    nii_path = os.path.join(base_dir, '%s.nii' % subject)
    gz_path = os.path.join(base_dir, '%s.nii.gz' % subject)
    return nii_path if os.path.exists(nii_path) else gz_path
 

class VesselRandomSampleAsync(object):
    
    def __init__(self, subject, sample_num, stage='train', async_load=True):
        
        self.use_rotation = cfg.TRAIN.DATA.USE_ROTATION if 'USE_ROTATION' in cfg.TRAIN.DATA.keys() else False
        self.use_skeleton = cfg.TRAIN.DATA.USE_SKELETON if 'USE_SKELETON' in cfg.TRAIN.DATA.keys() else False
        self.use_heatmap = cfg.TRAIN.DATA.USE_HEATMAP if 'USE_HEATMAP' in cfg.TRAIN.DATA.keys() else False
        self.axis = cfg.TRAIN.DATA.AXIS if 'AXIS' in cfg.TRAIN.DATA.keys() else 'z'
        self.stage = stage
        
        self.subject = subject
        if stage == 'train':
            self.image_proxy = nib.load(get_nii_path(cfg.TRAIN.DATA.NII_FOLDER, subject))
            self.mask_proxy = nib.load(get_nii_path(cfg.TRAIN.DATA.BLOOD_FOLDER, subject))
        else:
            self.image_proxy = nib.load(get_nii_path(cfg.TEST.DATA.NII_FOLDER, subject))
            test_mask_file = get_nii_path(cfg.TEST.DATA.BLOOD_FOLDER, subject)
            if os.path.exists(test_mask_file):
                self.mask_proxy = nib.load(test_mask_file)
            else:
                print('{} not exists.'.format(test_mask_file))
                self.mask_proxy = None

        if self.use_heatmap:
            self.heatmap_proxy = nib.load(get_nii_path(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '../heatmap'), subject))
        
        if self.use_skeleton and self.stage == 'train':
            self.skeleton_proxy = nib.load(get_nii_path(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '../skeleton_mask'), subject))
        
        if self.stage == 'train' and async_load:
            ex = ThreadPoolExecutor(max_workers=1)
            ex.submit(self.load_data, self)
        else:
            self.load_data(self)
            
            ###padding#########
            patch_size = cfg.TRAIN.DATA.PATCH_SIZE if stage == 'train' else cfg.TEST.DATA.PATCH_SIZE
            px, py, pz = self.reshape(patch_size, self.axis, True) 
            rpx = max(0, px - self.image.shape[0])
            rpy = max(0, py - self.image.shape[1])
            rpz = max(0, pz - self.image.shape[2])
            self.image = np.pad(self.image, ((0, rpx), (0, rpy), (0, rpz)), 'constant', constant_values=0)
            self.mask = np.pad(self.mask, ((0, rpx), (0, rpy), (0, rpz)), 'constant', constant_values=0)
            if self.use_heatmap:
                self.heatmap = np.pad(self.heatmap, ((0, rpx), (0, rpy), (0, rpz)), 'constant', constant_values=0)
            if self.use_skeleton and self.stage == 'train':
                self.skeleton = np.pad(self.skeleton, ((0, rpx), (0, rpy), (0, rpz)), 'constant', constant_values=255)
        
        if stage == 'train':
            self.num = sample_num
        else:
            self.coords = get_patch_coords(cfg.TEST.DATA.PATCH_SIZE, 
                                                         self.reshape(self.mask.shape, self.axis),
                                                         2)
            self.num = len(self.coords)
        
        self.sample_idx = 0
        self.wl, self.ww = cfg.TRAIN.DATA.WL_WW
    
    def __len__(self):
        return self.num
    
    @staticmethod
    def load_data(ss):
        ss.image = ss.image_proxy.get_data()
        if ss.mask_proxy is None:
            ss.mask = np.zeros_like(ss.image)
        else:
            ss.mask = ss.mask_proxy.get_data()
        
        if ss.use_heatmap:
            ss.heatmap = ss.heatmap_proxy.get_data()
        if ss.use_skeleton and ss.stage == 'train':
            ss.skeleton = ss.skeleton_proxy.get_data()
        
    @staticmethod
    def reshape(shape, axis, inv=False):
        if axis == 'y':
            if inv:
                shape = (shape[2], shape[0], shape[1])
            else:
                shape = (shape[1], shape[2], shape[0])
        elif axis == 'z':
            if inv:
                shape = (shape[1], shape[2], shape[0])
            else:
                shape = (shape[2], shape[0], shape[1])
        return shape
    
    @staticmethod
    def reshape_array(array, axis, inv=False):
        if axis == 'y':
            if inv:
                array = np.transpose(array, (2, 0, 1))
            else:
                array = np.transpose(array, (1, 2, 0))
        elif axis == 'z':
            if inv:
                array = np.transpose(array, (1, 2, 0))
            else:
                array = np.transpose(array, (2, 0, 1))
        return array
        
    def volume_size(self):
        return self.reshape(self.mask.shape, self.axis)
    
    def volume_mask(self):
        return self.reshape_array(self.mask, self.axis)
    
    def volume_heatmap(self):
        heatmap = (((self.heatmap - 1) / 4.0 + 1) * (self.mask > 0)).astype('float32')
        return self.reshape_array(heatmap, self.axis)
    
    def sample_validate(self):
        
        self.sample_idx = (self.sample_idx + 1) % self.num
        px, py, pz = cfg.TEST.DATA.PATCH_SIZE
        sx, sy, sz = self.coords[self.sample_idx]
        ex, ey, ez = sx + px, sy + py, sz + pz
        
        isx, isy, isz = self.reshape((sx, sy, sz), self.axis, True)
        iex, iey, iez = self.reshape((ex, ey, ez), self.axis, True)
        
        img = self.image[isx:iex, isy:iey, isz:iez]
        msk = self.mask[isx:iex, isy:iey, isz:iez]
        
        img = self.reshape_array(img, self.axis)
        msk = self.reshape_array(msk, self.axis)
        
        img = set_window_wl_ww(img, self.wl, self.ww)
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')[np.newaxis, ...]
        ddict = {'img': img, 'gt': msk, 'coord': np.array([[sx, ex],[sy, ey],[sz, ez]], 'int32')}
                        
        
        if self.use_heatmap:
            hp = self.heatmap[isx:iex, isy:iey, isz:iez]
            hp = self.reshape_array(hp, self.axis)
            hp = ((hp - 1) / 4.0 + 1) * (msk > 0)
            ddict['hp'] = hp[np.newaxis, ...].astype('float32')
        
        return ddict
    
    def sample_train(self):

        self.sample_idx = (self.sample_idx + 1) % self.num

        vx, vy, vz = self.reshape(self.mask_proxy.dataobj.shape, self.axis)
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE

        sx = random.randint(0, max(0, vx - px))
        sy = random.randint(0, max(0, vy - py))
        sz = random.randint(0, max(0, vz - pz))
        
        ex, ey, ez = sx + px,  sy + py, sz + pz
        sx, sy, sz = self.reshape((sx, sy, sz), self.axis, True)
        ex, ey, ez = self.reshape((ex, ey, ez), self.axis, True)
        
        if self.image_proxy.in_memory:
            img = self.image[sx:ex, sy:ey, sz:ez]
        else:
            img = self.image_proxy.slicer[sx:ex, sy:ey, sz:ez].get_data()
        
        if self.mask_proxy.in_memory:
            msk = self.mask[sx:ex, sy:ey, sz:ez]
        else:
            msk = self.mask_proxy.slicer[sx:ex, sy:ey, sz:ez].get_data()
            
        if msk.sum() < 100:
            return self.sample_train()
        
        img = self.reshape_array(img, self.axis)
        msk = self.reshape_array(msk, self.axis)
        
        #padding
        rpx = max(0, px - img.shape[0])
        rpy = max(0, py - img.shape[1])
        rpz = max(0, pz - img.shape[2])
        img = np.pad(img, ((0, rpx), (0, rpy), (0, rpz)), 'constant', constant_values=0)
        msk = np.pad(msk, ((0, rpx), (0, rpy), (0, rpz)), 'constant', constant_values=0)
        
        img = set_window_wl_ww(img, self.wl, self.ww)
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')[np.newaxis, ...]

        flip_y = np.random.choice(2)*2 - 1
        flip_z = np.random.choice(2)*2 - 1
        img = img[..., ::flip_y, ::flip_z].copy()
        msk = msk[..., ::flip_y, ::flip_z].copy()

        #ddict = {'img': img, 'gt':msk, 'subject': self.subject, 'coord': np.array([[x, x+px],[y, y+py],[z, z+pz]], 'int32')}
        ddict = {'img': img, 'gt':msk}
        if self.use_skeleton:
            if self.skeleton_proxy.in_memory:
                skel = self.skeleton[sx:ex, sy:ey, sz:ez]
            else:
                skel = self.skeleton_proxy.slicer[sx:ex, sy:ey, sz:ez].get_data()
            
            skel = self.reshape_array(skel, self.axis)
            skel = np.pad(skel, ((0, rpx), (0, rpy), (0, rpz)), 'constant', constant_values=255)
            
            skel = skel[..., ::flip_y, ::flip_z].copy()
            ddict['skel'] = skel

        if self.use_heatmap:
            if self.heatmap_proxy.in_memory:
                hp = self.heatmap[sx:ex, sy:ey, sz:ez]
            else:
                hp = self.heatmap_proxy.slicer[sx:ex, sy:ey, sz:ez].get_data()
            
            hp = self.reshape_array(hp, self.axis)
            hp = np.pad(hp, ((0, rpx), (0, rpy), (0, rpz)), 'constant', constant_values=0)
            
            hp = hp[..., ::flip_y, ::flip_z].copy()
            hp = ((hp - 1) / 4.0 + 1) * (msk > 0)
            ddict['hp'] = hp[np.newaxis, ...].astype('float32')

        return ddict
        
class VesselRandomSampler(object):
    
    def __init__(self, subject, sample_num, stage='train'):
        self.use_rotation = cfg.TRAIN.DATA.USE_ROTATION if 'USE_ROTATION' in cfg.TRAIN.DATA.keys() else False
        self.use_skeleton = cfg.TRAIN.DATA.USE_SKELETON if 'USE_SKELETON' in cfg.TRAIN.DATA.keys() else False
        self.use_heatmap = cfg.TRAIN.DATA.USE_HEATMAP if 'USE_HEATMAP' in cfg.TRAIN.DATA.keys() else False
        self.use_mww = False
        
        #t0 = time.time()
        self.subject = subject
        img_nii = nib.load(get_nii_path(cfg.TRAIN.DATA.NII_FOLDER, subject))
        image = img_nii.get_data()
        mask = nib.load(get_nii_path(cfg.TRAIN.DATA.BLOOD_FOLDER, subject)).get_data()
        #mask = mask.astype('bool').astype('uint8')
        spacing = img_nii.header['pixdim'][1:4]
        
        self.image = np.transpose(image, (2, 0, 1)) 
        self.mask = np.transpose(mask, (2, 0, 1))
        self.spacing = np.roll(spacing, 1, axis=0)
        #print(subject, 'load data need %.4f' % (time.time() - t0))
        
        ##############load heatmap#####################
        if self.use_heatmap:
            heatmap = nib.load(get_nii_path(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '../heatmap'), subject)).get_data().astype('uint8')
            self.heatmap = np.transpose(heatmap, (2, 0, 1))
        else:
            self.heatmap = np.zeros_like(self.mask)
        #t0 = time.time()
        ##############load skeleton####################
        if self.use_skeleton and stage == 'train':
            mat = scio.loadmat(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '../skeleton/%s.mat' % subject))
            self.points = np.roll(mat['points'], 1, axis=1)
            self.points = list(filter(lambda x: self.mask[x[0], x[1], x[2]] == 2, self.points)) #thin vessel skeleton points
            
            self.bifurcations = np.roll(mat['bifurcation'], 1, axis=1)
        else:
            self.points, self.bifurcations = [], []
        #print(subject, 'load skeleton need %.4f' % (time.time() - t0))
        
        #t0 = time.time()
        if img_nii.header['pixdim'][0] > 0:
            self.image = self.image[::-1, ...].copy()
            self.mask = self.mask[::-1, ...].copy()
            self.heatmap = self.heatmap[::-1, ...].copy()
            self.points = list(map(lambda x: np.array([self.mask.shape[0]-1-x[0],x[1],x[2]], 'int32'), self.points))
            self.bifurcations = list(map(lambda x: np.array([self.mask.shape[0]-1-x[0],x[1],x[2]], 'int32'), self.bifurcations))
        #print(subject, 'change direction need %.4f' % (time.time() - t0))
        
        #t0 = time.time()
        ########normalize###############################
        if self.spacing[0] != 0.5:
            x, y, z = self.image.shape
            shape = (int(x*self.spacing[0]/0.5), y, z)
            self.image = tensor_resize(self.image,  shape, False)
            self.mask = tensor_resize(self.mask, shape, True)
            if  self.use_heatmap:
                self.heatmap = tensor_resize(self.heatmap, shape, True)
            self.points = list(map(lambda x: np.array([x[0]*self.spacing[0]/0.5, x[1], x[2]]), self.points))
            self.bifurcations = list(map(lambda x: np.array([x[0]*self.spacing[0]/0.5, x[1], x[2]]), self.bifurcations))
        #print(subject, 'normalize slice spacing need %.4f' % (time.time() - t0))
        
        #t0 = time.time()
        #########rotation################################
        if self.use_rotation and stage == 'train':
            R_x = (np.random.choice(2) * 2 - 1) * np.random.choice(15)
            R_y = (np.random.choice(2) * 2 - 1) * np.random.choice(15)
            R_z = (np.random.choice(2) * 2 - 1) * np.random.choice(15)
            tt = getTransform(self.image.shape, R_x, R_y, R_z, (1.0, 1.0, 1.0))
            self.image = transform_image(tt, self.image, 'linear')
            self.mask = transform_image(tt, self.mask)
            if  self.use_heatmap:
                self.heatmap = transform_image(tt, self.heatmap)
            
            tt_inv = tt.GetInverse()
            self.points = list(map(lambda x: transform_point(tt_inv, np.array([x[0], x[1], x[2]],'float64')),self.points))
            self.bifurcations = list(map(lambda x: transform_point(tt_inv, np.array([x[0], x[1], x[2]],'float64')),self.bifurcations))
            
        self.points = list(map(lambda x: np.array([int(x[0]+0.5),int(x[1]+0.5),int(x[2]+0.5)], 'int32'),self.points))
        self.bifurcations = list(map(lambda x: np.array([int(x[0]+0.5),int(x[1]+0.5),int(x[2]+0.5)], 'int32'),self.bifurcations))   
        #print(subject, 'rotation need %.4f' % (time.time() - t0))
        
        #padding
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE if stage == 'train' else cfg.TEST.DATA.PATCH_SIZE
        rpx = max(0, px - self.image.shape[0])
        rpy = max(0, py - self.image.shape[1])
        rpz = max(0, pz - self.image.shape[2])
        self.image = np.pad(self.image, ((0, rpx), (0, rpy), (0, rpz)), 'constant', constant_values=0)
        self.mask = np.pad(self.mask, ((0, rpx), (0, rpy), (0, rpz)), 'constant', constant_values=0)
        self.heatmap = np.pad(self.heatmap, ((0, rpx), (0, rpy), (0, rpz)), 'constant', constant_values=0)
        
        #t0 = time.time()
        if self.use_skeleton and stage == 'train':
            self.skeleton = np.ones_like(self.mask) * 255
            for point in self.points:
                x, y, z = point
                self.skeleton[x-1:x+2, y-1:y+2, z-1:z+2] = self.mask[x-1:x+2, y-1:y+2, z-1:z+2]
        #print(subject, 'skeleton loss need %.4f' % (time.time() - t0))
        
        if stage == 'train':
            self.num = sample_num
        else:
            self.coords = get_patch_coords(cfg.TEST.DATA.PATCH_SIZE, self.mask.shape, 2)
            self.num = len(self.coords)
 
        self.sample_idx = 0
        self.wl, self.ww = cfg.TRAIN.DATA.WL_WW
    
    def __len__(self):
        return self.num
    
    def volume_size(self):
        return self.mask.shape
    
    def volume_mask(self):
        return self.mask
    
    def volume_heatmap(self):
        #return self.heatmap
        return (((self.heatmap - 1) / 4.0 + 1) * (self.mask > 0)).astype('float32')
    
    def sample_validate(self):
        x, y, z = self.coords[self.sample_idx]
        self.sample_idx = (self.sample_idx + 1) % self.num
        
        px, py, pz = cfg.TEST.DATA.PATCH_SIZE
        img = self.image[x:x+px, y:y+py, z:z+pz]
        msk = self.mask[x:x+px, y:y+py, z:z+pz]
        
        if self.use_mww:
            a = set_window_wl_ww(img.copy(), 400, 1400)
            b = set_window_wl_ww(img.copy(), 300, 800)
            c = set_window_wl_ww(img.copy(), 200, 500)
            img = np.concatenate((a[np.newaxis, ...], b[np.newaxis, ...], c[np.newaxis, ...]))    
            img = (img / 255.0) * 2.0 - 1.0
            img = img.astype('float32')
        else:
            img = set_window_wl_ww(img, self.wl, self.ww)
            img = (img / 255.0) * 2.0 - 1.0
            img = img.astype('float32')[np.newaxis, ...]
        
        ddict = {'img': img, 'gt': msk, 'coord': np.array([[x, x+px],[y, y+py],[z, z+pz]], 'int32')}
        
        if self.use_heatmap:
            hp = self.heatmap[x:x+px, y:y+py, z:z+pz]
            hp = ((hp - 1) / 4.0 + 1) * (msk > 0)
            ddict['hp'] = hp[np.newaxis, ...].astype('float32')
        
        return ddict
        
    
    def sample_train(self):
        
        self.sample_idx = (self.sample_idx + 1) % self.num
        
        vx, vy, vz = self.mask.shape
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE
        
        #c = np.random.choice([1,2], 1, p=[0.34, 0.66])[0]
        c = 2
        
        if c == 2:
            x = random.randint(0, max(0, vx - px))        
            y = random.randint(0, max(0, vy - py))
            z = random.randint(0, max(0, vz - pz))
        else:
            k = np.random.choice(list(range(len(self.bifurcations))), 1)[0]
            x, y, z = self.bifurcations[k]
            x = random.randint(min(max(0,  x-3*px//4), vx-px), min(max(0, x-px//4), vx-px))
            y = random.randint(max(0, min(vy, y+32) - py), min(max(0, y-32), vy - py))
            z = random.randint(max(0, min(vz, z+32) - pz), min(max(0, z-32), vz - pz))

        img = self.image[x:x+px, y:y+py, z:z+pz]
        msk = self.mask[x:x+px, y:y+py, z:z+pz]
        if msk.sum() < 100:
            return self.sample_train()
        
        if self.use_mww:
            a = set_window_wl_ww(img.copy(), 400, 1400)
            b = set_window_wl_ww(img.copy(), 300, 800)
            c = set_window_wl_ww(img.copy(), 200, 500)
            img = np.concatenate((a[np.newaxis, ...], b[np.newaxis, ...], c[np.newaxis, ...]))
            img = (img / 255.0) * 2.0 - 1.0
            img = img.astype('float32')
        else:
            img = set_window_wl_ww(img, self.wl, self.ww)
            img = (img / 255.0) * 2.0 - 1.0
            img = img.astype('float32')[np.newaxis, ...]
        
        flip_y = np.random.choice(2)*2 - 1
        flip_z = np.random.choice(2)*2 - 1
        img = img[..., ::flip_y, ::flip_z].copy()
        msk = msk[..., ::flip_y, ::flip_z].copy()
        
        #ddict = {'img': img, 'gt':msk, 'subject': self.subject, 'coord': np.array([[x, x+px],[y, y+py],[z, z+pz]], 'int32')}
        ddict = {'img': img, 'gt':msk}
        if self.use_skeleton:
            skel = self.skeleton[x:x+px, y:y+py, z:z+pz]
            skel = skel[..., ::flip_y, ::flip_z].copy()
            ddict['skel'] = skel
        
        if self.use_heatmap:
            hp = self.heatmap[x:x+px, y:y+py, z:z+pz]
            hp = hp[..., ::flip_y, ::flip_z].copy()
            hp = ((hp - 1) / 4.0 + 1) * (msk > 0)
            ddict['hp'] = hp[np.newaxis, ...].astype('float32')
        
        return ddict

class VesselVolumeSampler(object):
    
    def __init__(self, subject, sample_num, stage='train'):
        
        self.use_skeleton = cfg.TRAIN.DATA.USE_SKELETON if 'USE_SKELETON' in cfg.TRAIN.DATA.keys() else False
        self.subject = subject
        image = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '%s.nii.gz' % subject)).get_data()
        mask = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER, '%s.nii.gz' % subject)).get_data()
        mask = mask.astype('bool').astype('uint8')
        
        self.image = np.transpose(image, (2, 0, 1)) 
        self.mask = np.transpose(mask, (2, 0, 1))
        
        ##############load skeleton####################
        if self.use_skeleton and stage == 'train':
            points = scio.loadmat(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '../skeleton/%s.mat' % subject))['points']
            self.points = np.roll(points, 1, axis=1)
        else:
            self.points = []
        
        if stage == 'train':
            self.num = sample_num
        else:
            self.num = 1
 
        self.sample_idx = 0
        self.wl, self.ww = cfg.TRAIN.DATA.WL_WW
        self.image = set_window_wl_ww(self.image, self.wl, self.ww)
        self.image = (self.image / 255.0) * 2.0 - 1.0
        
    def __len__(self):
        return self.num
    
    def volume_size(self):
        return self.mask.shape
    
    def volume_mask(self):
        return self.mask
    
    def sample_validate(self):
        
        self.sample_idx = (self.sample_idx + 1) % self.num
                
        img = self.image.astype('float32')[np.newaxis, ...]
        msk = self.mask        
        ddict = {'img': img, 'gt': msk, 'coord': np.array([[0, msk.shape[0]],[0, msk.shape[1]],[0, msk.shape[2]]], 'int32')}
               
        return ddict
        
    
    def sample_train(self):
        
        self.sample_idx = (self.sample_idx + 1) % self.num
        
        img, msk, skel = self.random_rotation()
        img = img.astype('float32')[np.newaxis, ...]
        
        flip_y = np.random.choice(2)*2 - 1
        flip_z = np.random.choice(2)*2 - 1
        img = img[..., ::flip_y, ::flip_z].copy()
        msk = msk[..., ::flip_y, ::flip_z].copy()
        
        ddict = {'img': img, 'gt':msk, 'subject': self.subject}
        
        if self.use_skeleton:
            skel = skel[..., ::flip_y, ::flip_z].copy()
            ddict['skel'] = skel
        
        return ddict
    
    def random_rotation(self):
        
        R_x = (np.random.choice(2) * 2 - 1) * np.random.choice(15)
        R_y = (np.random.choice(2) * 2 - 1) * np.random.choice(15)
        R_z = (np.random.choice(2) * 2 - 1) * np.random.choice(15)
        
        tt = getTransform(self.image.shape, R_x, R_y, R_z, (1.0, 1.0, 1.0))
        image = transform_image(tt, self.image, 'linear')
        mask = transform_image(tt, self.mask)
        #mask = transform_image(tt, self.mask*255, 'linear')
        #mask = (mask > 0).astype('uint8')
        
        tt_inv = tt.GetInverse()
        points = list(map(lambda x: transform_point(tt_inv, np.array([x[0], x[1], x[2]],'float64')),self.points))
        points = list(map(lambda x: np.array([int(x[0]+0.5),int(x[1]+0.5),int(x[2]+0.5)], 'int32'), points))
        
        skeleton = np.ones_like(mask) * 255
        for point in self.points:
            x, y, z = point
            skeleton[x-1:x+2, y-1:y+2, z-1:z+2] = mask[x-1:x+2, y-1:y+2, z-1:z+2]
        
        return image, mask, skeleton

class VesselSDFSampler(object):
    
    def __init__(self, subject, sample_num, stage='train'):
        
        img_nii = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '%s.nii.gz' % subject))
        image = img_nii.get_data()
        
        data = scio.loadmat(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '../sdf_zc/%s.mat' % subject))
        sdf = data['sdf'].astype('float32') / 32767
        mask = np.concatenate((sdf[np.newaxis, ...], data['grads'].astype('float32')))
        
        spacing = img_nii.header['pixdim'][1:4]
        self.image = np.transpose(image, (2, 0, 1)) 
        self.mask = np.transpose(mask, (0, 3, 1, 2))
        self.spacing = np.roll(spacing, 1, axis=0)
        
        '''
        if self.spacing[0] != 0.5:
            x, y, z = self.image.shape
            shape = (int(x*self.spacing[0]/0.5), y, z)
            self.image = tensor_resize(self.image,  shape, False)
            self.mask = tensor_resize(self.mask, shape, False)
        
        if img_nii.header['pixdim'][0] > 0:
            self.image = self.image[::-1, ...]
            self.mask = self.mask[::-1, ...]
        '''
        
        if stage == 'train':
            self.num = sample_num
        else:
            self.coords = get_patch_coords(cfg.TEST.DATA.PATCH_SIZE, self.mask.shape[-3:], 2)
            self.num = len(self.coords)
 
        self.sample_idx = 0
        self.wl, self.ww = cfg.TRAIN.DATA.WL_WW
    
    def __len__(self):
        return self.num
    
    def volume_size(self):
        return self.mask.shape
    
    def volume_mask(self):
        return self.mask
    
    def sample_validate(self):
        x, y, z = self.coords[self.sample_idx]
        self.sample_idx = (self.sample_idx + 1) % self.num
        
        px, py, pz = cfg.TEST.DATA.PATCH_SIZE
        img = self.image[x:x+px, y:y+py, z:z+pz]
        msk = self.mask[:, x:x+px, y:y+py, z:z+pz]
        
        img = set_window_wl_ww(img, self.wl, self.ww)
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')[np.newaxis, ...]
        
        return {'img': img, 'gt': msk, 'coord': np.array([[x, x+px],[y, y+py],[z, z+pz]], 'int32')}
    
    def sample_train(self):
        
        self.sample_idx = (self.sample_idx + 1) % self.num
        
        vx, vy, vz = self.image.shape
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE
      
        x = random.randint(0, vx - px)        
        y = random.randint(0, vy - py)
        z = random.randint(0, vz - pz)
        
        img = self.image[x:x+px, y:y+py, z:z+pz]
        msk = self.mask[:, x:x+px, y:y+py, z:z+pz]
        
        img = set_window_wl_ww(img, self.wl, self.ww)
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')[np.newaxis, ...]
        
        flip_y = np.random.choice(2)*2 - 1
        flip_z = np.random.choice(2)*2 - 1
        img = img[..., ::flip_y, ::flip_z].copy()
        msk = msk[..., ::flip_y, ::flip_z].copy()
        
        return {'img': img, 'gt': msk}
    
def sample_subjects(subjects,  patch_num):
    data_lst = []
    for subject in subjects:
        sampler = VesselPatchSampler(subject)
        data_lst.extend(sampler.sample(patch_num))
    return data_lst

class Patch3DLoader(data.Dataset):
    
    def __init__(self, data):
        self.data = data
        self.flip = False
    
    def __getitem__(self, index):
        
        img, gt = self.data[index]
        img = (img / 255.0) * 2.0 - 1.0
        if self.flip:
            flip_y = np.random.choice(2)*2 - 1
            flip_z = np.random.choice(2)*2 - 1
            img = img[:, ::flip_y, ::flip_z]
            gt = gt[:, ::flip_y, ::flip_z]
        
        img = img.astype('float32')
        img = img[np.newaxis, ...]
        img, gt = torch.from_numpy(img), torch.from_numpy(gt)
        
        return {'img': img, 'gt': gt}
    
    def __len__(self):
        return len(self.data)

class VesselDatasetSampler(object):
    
    def __init__(self, train_lst):
        with open(train_lst, 'r') as f:
            lines = f.readlines()
            self.subjects = [line.strip() for line in lines]
        
        self.ex = None
        self.objs = []
    
    def asyn_sample(self, subject_num=100, patch_num=100):
        
        subject_num = min(subject_num, len(self.subjects))
        samples = np.random.choice(range(len(self.subjects)), size=subject_num, replace= False)
        subjects = [self.subjects[idx] for idx in samples]
        self.ex = ProcessPoolExecutor(max_workers=20)
        self.objs = []
        step = int(math.ceil((subject_num / 20)))
        monitor = progress_monitor(total=len(range(0, len(subjects), step)))
        for i in range(0, len(subjects), step):
            future  = self.ex.submit(sample_subjects, subjects[i:i+step], patch_num)
            future.add_done_callback(fn=monitor)
            self.objs.append(future)
        print('data processing in async ...')
    
    def get_data_loader(self):
        self.ex.shutdown(wait=True)
        data_lst = []
        for obj in self.objs:
            data_lst.extend(obj.result())
        return Patch3DLoader(data_lst)
                         
class VesselDS(BaseDataset):
    def __init__(self, para_dict, stage):
        super(VesselDS, self).__init__(para_dict, stage)

    def train_init(self):
        self.train_list = self.para_dict.get("train_list", None)
        with open(self.train_list, 'r') as f:
            lines = f.readlines()
            self.npz_files = [line.strip() for line in lines]
        #random.shuffle(self.npz_files)
        
        self.sample = 2
        
        self.num = len(self.npz_files) * self.sample
        
        self.rotate_transform = False
        self.flip = True
        
        self.use_sdm = cfg.TRAIN.DATA.USE_SDM
        
    
    def crop_cardiac(self, flip_z=True):
        x0, y0, z0, w, h, d = self.para_dict['cardiac_bbox'][self.subject]
        x1, y1, z1 = x0+w-1, y0+h-1, z0+d-1
        
        if flip_z:
            z0, z1 = self.img.shape[0]-1-z1, self.img.shape[0]-z0-1
        
        x0, x1 = max(0, x0 - 32), min(self.img.shape[1]-1, x1+32)
        y0, y1 = max(0, y0 - 32), min(self.img.shape[2]-1, y1+32)
        z0, z1 = max(0, z0 - 48), min(self.img.shape[0]-1, z1+16)
        if (x1 - x0 + 1) < self.patch_size[1]:
            x0 = min(self.img.shape[1] - self.patch_size[1],
                     max(0, x0 - (self.patch_size[1] - (x1-x0+1)) // 2))
            x1 = x0 + self.patch_size[1]-1
                
        if (y1 - y0 + 1) < self.patch_size[2]:
            y0 = min(self.img.shape[2] - self.patch_size[2],
                     max(0, y0 - (self.patch_size[2] - (y1-y0+1)) // 2))
            y1 = y0 + self.patch_size[2]-1
                
        coords = get_patch_coords(self.patch_size, (z1-z0+1, x1-x0+1, y1-y0+1), 2)
        self.coords = [(c[0]+z0, c[1]+x0, c[2]+y0) for c in coords]
        
    def val_init(self):
        '''
        self.val_list = self.para_dict.get("val_list", None)
        with open(self.val_list, 'r') as f:
            lines = f.readlines()
            self.list = [line.strip() for line in lines]
        
        self.val_inited = False
        self.data_lst = []
        self.ex = ThreadPoolExecutor()
        self.objs = []
        for subject in self.list:
            self.objs.append(self.ex.submit(load_subject_validate, subject))
        
        self.ex.shutdown(wait=True)
        for obj in self.objs:
            self.data_lst.extend(obj.result())
        self.val_inited = True
        self.num = len(self.data_lst)                
        '''
        self.use_volume = False
        if 'subject' in self.para_dict:
            self.subject = self.para_dict.get("subject", None)
            img_nii = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER,
                                           '%s.nii.gz' % self.subject))
            self.spacing = img_nii.header['pixdim'][1:4]
            img = img_nii.get_data()
            msk = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER,
                                        '%s.nii.gz' % self.subject)).get_data()
            
            self.img = np.transpose(img, (2, 0, 1))
            self.msk = np.transpose(msk, (2, 0, 1))
  
            if img_nii.header['pixdim'][0] > 0:
                self.img = self.img[::-1, ...]
                self.msk = self.msk[::-1, ...]
                
            self.spacing = np.roll(self.spacing, 1, axis=0)
            
            WL, WW = cfg.TRAIN.DATA.WL_WW
            self.img = set_window_wl_ww(self.img, WL, WW)
            self.msk[self.msk > 0] = 1
            
            self.patch_size = cfg.TEST.DATA.PATCH_SIZE
            if 'cardiac_bbox' in self.para_dict:
                self.crop_cardiac(img_nii.header['pixdim'][0] > 0)
            else:
                self.coords = get_patch_coords(self.patch_size, self.img.shape, 2)
            
            self.num = len(self.coords)
            self.use_volume = True
            
            
        else:
            self.val_npz = self.para_dict.get("val_list", None)
            self.data_lst = np.load(self.val_npz)['data']
            self.num = len(self.data_lst)
        
    def test_init(self):
        self.subject = self.para_dict.get("subject", None)
        img_nii = nib.load(os.path.join(cfg.TEST.DATA.NII_FOLDER,
                                    '%s.nii.gz' % self.subject))
        img = img_nii.get_data()
        
        self.img = np.transpose(img, (2, 0, 1))
        self.img = self.img[::-1, ...]
        
        WL, WW = cfg.TRAIN.DATA.WL_WW
        self.img = set_window_wl_ww(self.img, WL, WW)
        self.patch_size = cfg.TEST.DATA.PATCH_SIZE#(32, 512, 512)  # PATCH_SIZE
        if 'cardiac_bbox' in self.para_dict:
            self.crop_cardiac(img_nii.header['pixdim'][0] > 0)
        else:
            self.coords = get_patch_coords(self.patch_size, self.img.shape, 2)
        self.num = len(self.coords)

    def train_load(self, index):
        npz_pth = self.npz_files[int(index/self.sample)]
        npz = np.load(npz_pth)
        
        img, gt = npz['img'], npz['gt']
        
        
        if self.rotate_transform:
            R_x = (np.random.choice(2) * 2 - 1)*np.random.choice(15)
            tt = getTransform(x.shape, R_x, 0, 0)
            img = transform_image(tt, img, 'linear')
            gt = transform_image(tt, gt)
               
        x1, y1, z1 = img.shape
        x0, y0, z0 = cfg.TRAIN.DATA.PATCH_SIZE

        x = random.randint(0, x1 - x0)
        y = random.randint(0, y1 - y0)
        z = random.randint(0, z1 - z0)
    
        img = img[x:x + x0, y:y + y0, z:z + z0]
        gt = gt[x:x + x0, y:y + y0, z:z + z0]
        
        if self.flip:
            f_x, f_y = np.random.choice(2)*2 - 1, np.random.choice(2)*2 - 1
            img = img[:, ::f_x, ::f_y].copy()
            gt = gt[:, ::f_x, ::f_y].copy()
        
        WL, WW = cfg.TRAIN.DATA.WL_WW
        img = set_window_wl_ww(img, WL, WW)
        img = (img / 255.0) * 2.0 - 1.0

        img = img.astype('float32')
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)
        gt = torch.from_numpy(gt)
        
        if self.use_sdm:
            sdm = npz['sdm']
            sdm = sdm[x:x + x0, y:y + y0, z:z + z0]
            if self.flip:
                sdm = sdm[:, ::f_x, ::f_y].copy()
            sdm = torch.from_numpy(sdm)
            
            return {'img': img, 'gt': gt, 'sdm': sdm}
        
        return {'img': img, 'gt': gt}

    def test_load(self, index):
        x, y, z = self.coords[index]

        img = self.img[x:x + self.patch_size[0],
              y:y + self.patch_size[1],
              z:z + self.patch_size[2]]

        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)

        coord = np.array([[x, x + self.patch_size[0]],
                          [y, y + self.patch_size[1]],
                          [z, z + self.patch_size[2]]])
        coord = torch.from_numpy(coord)

        return img, coord

    def val_load(self, index):
        if self.use_volume:
            x, y, z = self.coords[index]
            img = self.img[x:x + self.patch_size[0],
                           y:y + self.patch_size[1],
                           z:z + self.patch_size[2]]
            
            img = (img / 255.0) * 2.0 - 1.0
            img = img.astype('float32')
            img = img[np.newaxis, :, :, :]
            img = torch.from_numpy(img)
            
            coord = np.array([[x, x + self.patch_size[0]],
                              [y, y + self.patch_size[1]],
                              [z, z + self.patch_size[2]]])
            coord = torch.from_numpy(coord)
            
            return img, coord
        
        else:
            image, mask = self.data_lst[index]
            image = (image / 255.0) * 2.0 - 1.0
            image = image.astype('float32')
            image = image[np.newaxis, ...]
            mask = mask.astype('uint8')
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

            return {'img': image, 'gt': mask}

    def volume_size(self):
        return self.img.shape
    
    def volume_mask(self):        
        return self.msk
        
    
    '''
    def __len__(self):
        if self.stage == 'val':
            if not self.val_inited:
                self.ex.shutdown(wait=True)
                for obj in self.objs:
                    self.data_lst.extend(obj.result())
                self.val_inited = True
                
            return len(self.data_lst)
            
        else:
            return super().len()
    '''
    
    def save(self, seg, res_pth, prob=False):

        img = nib.load(os.path.join(cfg.TEST.DATA.NII_FOLDER, '%s.nii.gz' % self.subject))
        seg = seg[::-1, ...]
        seg = np.transpose(seg, (1, 2, 0))
        seg = seg.astype('float32' if prob else 'uint8')
        affine = img.affine
        seg_img = nib.Nifti1Image(seg, affine)
        nib.save(seg_img, res_pth)

class VesselSubjectDS2(BaseDataset):
    
    def __init__(self, para_dict, stage, VS=VesselRandomSampler):
        self.VS = VS
        super(VesselSubjectDS2, self).__init__(para_dict, stage)
        
    def train_init(self):
        
        self.subjects = self.para_dict.get("subjects", None)
        self.sample_num_per_subject = self.para_dict.get("sample", 64)        
        self.num = len(self.subjects) * self.sample_num_per_subject        
        #self.subject = ''
        #self.data_sampler = None
        self.cache_subjects = []
        self.cache_samplers = []
        self.max_cache_num = self.para_dict.get("cache_num", 10)      
    
    def train_load(self, index):
        
        subject = self.subjects[index // self.sample_num_per_subject]        
        if subject not in self.cache_subjects:
            self.cache_subjects.append(subject)
            self.cache_samplers.append(self.VS(subject, self.sample_num_per_subject))
            if len(self.cache_subjects) > self.max_cache_num:
                self.cache_subjects.pop(0)
                self.cache_samplers.pop(0)
                
        return self.cache_samplers[self.cache_subjects.index(subject)].sample_train()
        
        #if self.subject != subject:
        #    self.subject = subject
        #    self.data_sampler = self.VS(subject, self.sample_num_per_subject)        
        #return self.data_sampler.sample_train()
    
    def val_init(self):        
        self.subject = self.para_dict.get("subject", None)
        self.data_sampler = self.VS(self.subject, -1, stage='validate')
        self.num = len(self.data_sampler)
        
    def val_load(self, index):        
        return self.data_sampler.sample_validate()
    
    def volume_size(self):
        return self.data_sampler.volume_size()
    
    def volume_mask(self):
        return self.data_sampler.volume_mask()
    
    def volume_heatmap(self):
        return self.data_sampler.volume_heatmap()
    
    def save(self, seg, res_pth, prob=False):
        img_nii = nib.load(os.path.join(cfg.TEST.DATA.NII_FOLDER, '%s.nii.gz' % self.subject))
        orign_size = tuple(np.roll(img_nii.header['dim'][1:4], 1, 0))
        if orign_size != seg.shape[-3:]:
            seg = tensor_resize(seg, orign_size, False if prob else True)
        
        if img_nii.header['pixdim'][0] > 0:
            seg = seg[::-1, ...]
        seg = np.transpose(seg, (1, 2, 0))
        #seg = np.transpose(seg, (0, 2, 3, 1))
        seg = seg.astype('float32' if prob else 'uint8')
        affine = img_nii.affine
        seg_img = nib.Nifti1Image(seg, affine)
        nib.save(seg_img, res_pth)


class VesselSubjectDS(BaseDataset):
    
    def __init__(self, para_dict, stage, VS=VesselRandomSampleAsync):
        self.VS = VS
        super(VesselSubjectDS, self).__init__(para_dict, stage)
        
    def train_init(self):
        
        self.subjects = self.para_dict.get("subjects", None)
        self.sample_num_per_subject = self.para_dict.get("sample", 64)        
        self.num = len(self.subjects) * self.sample_num_per_subject
        self.cache_subjects = []
        self.cache_samplers = []
        self.max_cache_num = self.para_dict.get("cache_num", 10)
        self.subjects_per_worker = self.para_dict.get("subjects_per_worker", [])
        self.first_load = True
    
    def train_load(self, index):
        
        subject = self.subjects[index // self.sample_num_per_subject]
        
        if self.first_load:
            for ss in self.subjects_per_worker:
                if ss[0] == index // self.sample_num_per_subject:
                    self.subject_iter = iter(ss)
                    break
            self.ex = ThreadPoolExecutor(max_workers=1)
            self.job = self.ex.submit(lambda x: self.VS(x, self.sample_num_per_subject), 
                                              self.subjects[next(self.subject_iter)])
            self.first_load = False
       
        if subject not in self.cache_subjects:
            self.ex.shutdown(wait=True)
            self.cache_subjects.append(subject)
            self.cache_samplers.append(self.job.result())
            
            try:
                self.ex = ThreadPoolExecutor(max_workers=1)
                self.job = self.ex.submit(lambda x: self.VS(x, self.sample_num_per_subject), 
                                                  self.subjects[next(self.subject_iter)])
            except Exception as ex:
                1 + 1
                
            if len(self.cache_subjects) > self.max_cache_num:
                self.cache_subjects.pop(0)
                self.cache_samplers.pop(0) 
                
        return self.cache_samplers[self.cache_subjects.index(subject)].sample_train()
    
    def val_init(self):        
        self.subject = self.para_dict.get("subject", None)
        self.data_sampler = self.VS(self.subject, -1, stage='validate')
        self.num = len(self.data_sampler)
        
    def val_load(self, index):        
        return self.data_sampler.sample_validate()
    
    def volume_size(self):
        return self.data_sampler.volume_size()
    
    def volume_mask(self):
        return self.data_sampler.volume_mask()
    
    def volume_heatmap(self):
        return self.data_sampler.volume_heatmap()
    
    def save(self, seg, res_pth, prob=False):
        img_nii = nib.load(os.path.join(cfg.TEST.DATA.NII_FOLDER, '%s.nii.gz' % self.subject))
        vx, vy, vz = img_nii.dataobj.shape
        if img_nii.header['pixdim'][0] > 0:
            seg = seg[::-1, ...]

        axis = cfg.TRAIN.DATA.AXIS if 'AXIS' in cfg.TRAIN.DATA.keys() else 'z'
        seg = VesselRandomSampleAsync.reshape_array(seg, axis, True)
        seg = seg[:vx, :vy, :vz]
        #if not prob:
        #    seg = seg > 0
        seg = seg.astype('float32' if prob else 'uint8')
        affine = img_nii.affine
        seg_img = nib.Nifti1Image(seg, affine)
        nib.save(seg_img, res_pth)   
    