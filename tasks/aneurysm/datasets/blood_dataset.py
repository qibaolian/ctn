from utils.config import cfg
from tasks.aneurysm.datasets.base_dataset import BaseDataset
from tasks.aneurysm.datasets.base_dataset import BaseSubjectSampler, MemoryDatasetSampler, FileDatasetSampler
from tasks.aneurysm.datasets.data_utils import *
from tasks.aneurysm.rotation_3d import transform_image
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
import time
import h5py
import scipy
from scipy.ndimage import interpolation
import torch.nn.functional as F

from tasks.aneurysm.rotation_3d import getTransform, transform_point, transform_image

def tensor_resize(image, shape, is_label=True):
    x = torch.from_numpy(image[np.newaxis, np.newaxis, ...])
    if is_label:
        y = F.interpolate(x.float(), shape, mode='nearest')
    else:
        y = F.interpolate(x.float(), shape, mode= 'trilinear', align_corners=True)
    y = y.numpy().astype(image.dtype)[0, 0]
    return y

class VesselRandomSampler(object):
    
    def __init__(self, subject, sample_num, stage='train'):
        self.use_rotation = cfg.TRAIN.DATA.USE_ROTATION if 'USE_ROTATION' in cfg.TRAIN.DATA.keys() else False
        self.use_skeleton = cfg.TRAIN.DATA.USE_SKELETON if 'USE_SKELETON' in cfg.TRAIN.DATA.keys() else False
        self.use_heatmap = cfg.TRAIN.DATA.USE_HEATMAP if 'USE_HEATMAP' in cfg.TRAIN.DATA.keys() else False
        
        self.subject = subject
        img_nii = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '%s.nii.gz' % subject))
        image = img_nii.get_data()
        mask = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER, '%s.nii.gz' % subject)).get_data()
        spacing = img_nii.header['pixdim'][1:4]
        
        self.image, self.mask = np.transpose(image, (2, 0, 1)), np.transpose(mask, (2, 0, 1))        
        self.mask = self.mask.astype('bool').astype('uint8')
        self.spacing = np.roll(spacing, 1, axis=0)
        
        ##############load heatmap#####################
        if self.use_heatmap:
            heatmap = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '../heatmap/%s.nii.gz' % subject)).get_data().astype('float32')
            self.heatmap = np.transpose(heatmap, (2, 0, 1))
        else:
            self.heatmap = np.zeros_like(self.mask)
        
        ##############load skeleton####################
        if self.use_skeleton and stage == 'train':
            points = scio.loadmat(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '../skeleton/%s.mat' % subject))['points']
            self.points = np.roll(points, 1, axis=1)
        else:
            self.points = []
            
        ########normalize###############################
        if self.spacing[0] > 0.625:
            x, y, z = self.image.shape
            shape = (int(x*self.spacing[0]/0.625), y, z)
            self.image = tensor_resize(self.image,  shape, False)
            self.mask = tensor_resize(self.mask, shape, True)
            self.heatmap = tensor_resize(self.heatmap, shape, True)
            self.points = list(map(lambda x: np.array([int(x[0]*self.spacing[0]/0.625+0.5), x[1], x[2]], 'int32'), self.points))
        
        #########rotation################################
        if self.use_rotation and stage == 'train':
            R_x = (np.random.choice(2) * 2 - 1) * np.random.choice(10)
            R_y = (np.random.choice(2) * 2 - 1) * np.random.choice(10)
            R_z = (np.random.choice(2) * 2 - 1) * np.random.choice(10)
            tt = getTransform(self.image.shape, R_x, R_y, R_z, (1.0, 1.0, 1.0))
            self.image = transform_image(tt, self.image, 'linear')
            self.mask = transform_image(tt, self.mask)
            self.heatmap = transform_image(tt, self.heatmap, 'linear')
            
            tt_inv = tt.GetInverse()
            self.points = list(map(lambda x: transform_point(tt_inv, np.array([x[0], x[1], x[2]],'float64')),self.points))
            self.points = list(map(lambda x: np.array([int(x[0]+0.5),int(x[1]+0.5),int(x[2]+0.5)], 'int32'),self.points))
        
        self.skeleton = np.ones_like(self.mask) * 255
        if self.use_skeleton and stage == 'train':
            #self.skeleton = np.zeros_like(self.mask)
            #for point in points:
            #    x, y, z = point
            #    self.skeleton[x-1:x+2, y-1:y+2, z-1:z+2] = 1
            #self.skeleton = self.skeleton * self.mask
            #self.skeleton[self.skeleton == 0] = 255
            self.skeleton = np.ones_like(self.mask) * 255
            for point in self.points:
                x, y, z = point
                self.skeleton[x-1:x+2, y-1:y+2, z-1:z+2] = self.mask[x-1:x+2, y-1:y+2, z-1:z+2]
        
        if cfg.TRAIN.DATA.POSITION == 'Head':
            num = self.image.shape[0] // 3 + 64
            self.image, self.mask = self.image[-num:, ...], self.mask[-num:, ...] 
            self.heatmap, self.skeleton = self.heatmap[-num:, ...], self.skeleton[-num:, ...]
        elif cfg.TRAIN.DATA.POSITION == 'Neck':
            num = self.image.shape[0] // 3
            self.image, self.mask = self.image[0:num, ...], self.mask[0:num, ...]
            self.heatmap, self.skeleton = self.heatmap[:num,...], self.skeleton[:num, ...]
        
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
        return self.heatmap
    
    def sample_validate(self):
        
        x, y, z = self.coords[self.sample_idx]
        self.sample_idx = (self.sample_idx + 1) % self.num
        
        px, py, pz = cfg.TEST.DATA.PATCH_SIZE
        img = self.image[x:x+px, y:y+py, z:z+pz]
        msk = self.mask[x:x+px, y:y+py, z:z+pz]
        
        img = set_window_wl_ww(img, self.wl, self.ww)
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')[np.newaxis, ...]
        msk = msk.astype('bool').astype('uint8')
        
        ddict = {'img': img, 'gt': msk, 'coord': np.array([[x, x+px],[y, y+py],[z, z+pz]], 'int32')}
        
        if self.use_heatmap:
            hp = self.heatmap[x:x+px, y:y+py, z:z+pz]
            ddict['hp'] = hp[np.newaxis, ...]
        
        return ddict
    
    
    def sample_train(self):
        
        self.sample_idx = (self.sample_idx + 1) % self.num
        
        vx, vy, vz = self.mask.shape
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE
      
        x = random.randint(0, vx - px)        
        y = random.randint(0, vy - py)
        z = random.randint(0, vz - pz)
        
        img = self.image[x:x+px, y:y+py, z:z+pz]
        msk = self.mask[x:x+px, y:y+py, z:z+pz]
        
        img = set_window_wl_ww(img, self.wl, self.ww)
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')[np.newaxis, ...]
        msk = msk.astype('bool').astype('uint8')
        
        flip_y = np.random.choice(2)*2 - 1
        flip_z = np.random.choice(2)*2 - 1
        img = img[..., ::flip_y, ::flip_z].copy()
        msk = msk[..., ::flip_y, ::flip_z].copy()
        
        ddict = {'img': img, 'gt':msk, 'subject': self.subject, 'coord': np.array([[x, x+px],[y, y+py],[z, z+pz]], 'int32')}
        if self.use_skeleton:
            skel = self.skeleton[x:x+px, y:y+py, z:z+pz]
            skel = skel[..., ::flip_y, ::flip_z].copy()
            ddict['skel'] = skel
        
        if self.use_heatmap:
            hp = self.heatmap[x:x+px, y:y+py, z:z+pz]
            hp = hp[..., ::flip_y, ::flip_z].copy()
            ddict['hp'] = hp[np.newaxis, ...]
        
        return ddict

class VesselSubjectDS(BaseDataset):
    
    def __init__(self, para_dict, stage, VS=VesselRandomSampler):
        self.VS = VS
        super(VesselSubjectDS, self).__init__(para_dict, stage)
        
    def train_init(self):
        
        self.subjects = self.para_dict.get("subjects", None)
        self.sample_num_per_subject = self.para_dict.get("sample", 64)        
        self.num = len(self.subjects) * self.sample_num_per_subject        
        
        #self.subject = ''
        #self.data_sampler = None
        self.cache_subjects = []
        self.cache_samplers = []
        self.max_cache_num = self.para_dict.get("cache_num", 12)
    
    def train_load(self, index):
        subject = self.subjects[index // self.sample_num_per_subject]        
        if subject not in self.cache_subjects:
            self.cache_subjects.append(subject)
            self.cache_samplers.append(self.VS(subject, self.sample_num_per_subject))
            if len(self.cache_subjects) > self.max_cache_num:
                self.cache_subjects.pop(0)
                self.cache_samplers.pop(0)
                
        return self.cache_samplers[self.cache_subjects.index(subject)].sample_train()
        
        #subject = self.subjects[index // self.sample_num_per_subject]        
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
        if  img_nii.header['pixdim'][3] > 0.625:
            orign_size = (int(seg.shape[0]*0.625/mg_nii.header['pixdim'][3]), seg.shape[1], seg.shape[2])
            seg = tensor_resize(seg, orign_size, False if prob else True)
        
        seg = np.transpose(seg, (1, 2, 0))
        
        msk = np.zeros(img_nii.header['dim'][1:4], 'float32' if prob else 'uint8')
        if cfg.TEST.DATA.POSITION == 'Head':
            msk[:, :, -seg.shape[2]:] = seg
        elif cfg.TEST.DATA.POSITION == 'Neck':
            msk[:,:, 0:seg.shape[2]] = seg
        else:
            msk = seg
            
        affine = img_nii.affine
        seg_img = nib.Nifti1Image(msk, affine)
        nib.save(seg_img, res_pth)