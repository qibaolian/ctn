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

from tasks.aneurysm.rotation_3d import getTransform, transform_point, transform_image

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
        self.use_cascade = cfg.TRAIN.DATA.USE_CASCADE if 'USE_CASCADE' in cfg.TRAIN.DATA.keys() else False
        self.use_rotation = cfg.TRAIN.DATA.USE_ROTATION if 'USE_ROTATION' in cfg.TRAIN.DATA.keys() else False
        self.use_skeleton = cfg.TRAIN.DATA.USE_SKELETON if 'USE_SKELETON' in cfg.TRAIN.DATA.keys() else False
        self.use_heatmap = cfg.TRAIN.DATA.USE_HEATMAP if 'USE_HEATMAP' in cfg.TRAIN.DATA.keys() else False
        
        self.random = False
        
        self.subject = subject
        img_nii = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '%s.nii.gz' % subject))
        image = img_nii.get_data()
        mask = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER, '%s.nii.gz' % subject)).get_data()
        mask = mask.astype('bool').astype('uint8')
        spacing = img_nii.header['pixdim'][1:4]
        
        self.image = np.transpose(image, (2, 0, 1))
        self.mask = np.transpose(mask, (2, 0, 1))
        self.spacing = np.roll(spacing, 1, axis=0)

        ############## load cascade #####################
        if self.use_cascade:
            feature_map = nib.load(
                os.path.join(cfg.TRAIN.DATA.CASCADE_FOLDER, '%s_seg.nii.gz' % subject)).get_data().astype('float32')
            self.feature_map = np.transpose(feature_map, (2, 0, 1))

        ##############load heatmap#####################
        if self.use_heatmap:
            heatmap = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '../heatmap/%s.nii.gz' % subject)).get_data().astype('float32')
            self.heatmap = np.transpose(heatmap, (2, 0, 1))
        else:
            self.heatmap = np.zeros_like(self.mask)
        
        ##############load skeleton####################
        if self.use_skeleton and stage == 'train':
            points = scio.loadmat(os.path.join(cfg.TRAIN.DATA.SKELETON_FOLDER, '%s.mat' % subject))['points']
            self.points = np.roll(points, 1, axis=1)
            self.skeleton = np.zeros_like(self.mask)
        else:
            self.points = []
            
        #if img_nii.header['pixdim'][0] > 0:
        #    self.image = self.image[::-1, ...].copy()
        #    self.mask = self.mask[::-1, ...].copy()
        #    self.heatmap = self.heatmap[::-1, ...].copy()
        #    self.points = list(map(lambda x: np.array([self.mask.shape[0]-1-x[0],x[1],x[2]], 'int32'), self.points))
            
        ########normalize###############################
        if self.spacing[0] > 0.625:
            x, y, z = self.image.shape
            shape = (int(x*self.spacing[0]/0.625), y, z)
            self.image = tensor_resize(self.image,  shape, False)
            self.mask = tensor_resize(self.mask, shape, True)
            self.heatmap = tensor_resize(self.heatmap, shape, True)
            self.points = list(map(lambda x: np.array([int(x[0]*self.spacing[0]/0.625+0.5), x[1], x[2]], 'int32'), self.points))
            if self.use_cascade:
                self.feature_map = tensor_resize(self.feature_map, shape, True)
        
        #########rotation################################
        if self.use_rotation and stage == 'train':
            R_x = (np.random.choice(2) * 2 - 1) * np.random.choice(15)
            R_y = (np.random.choice(2) * 2 - 1) * np.random.choice(15)
            R_z = (np.random.choice(2) * 2 - 1) * np.random.choice(15)
            tt = getTransform(self.image.shape, R_x, R_y, R_z, (1.0, 1.0, 1.0))
            self.image = transform_image(tt, self.image, 'linear')
            self.mask = transform_image(tt, self.mask)
            self.heatmap = transform_image(tt, self.heatmap, 'linear')
            if self.use_cascade:
                self.feature_map = transform_image(tt, self.feature_map, 'linear')
            
            tt_inv = tt.GetInverse()
            self.points = list(map(lambda x: transform_point(tt_inv, np.array([x[0], x[1], x[2]],'float64')),self.points))
            self.points = list(map(lambda x: np.array([int(x[0]+0.5),int(x[1]+0.5),int(x[2]+0.5)], 'int32'),self.points))            
            
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
        
        if stage == 'train':
            self.num = sample_num
            self.coords = get_patch_coords(cfg.TRAIN.DATA.PATCH_SIZE, self.mask.shape, 2)
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
                    
    def sample_random_coord(self):
        
        vx, vy, vz = self.mask.shape
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE
      
        x = random.randint(0, vx - px)        
        y = random.randint(0, vy - py)
        z = random.randint(0, vz - pz)
        
        return x, y, z
    
    def sample_validate(self):
        x, y, z = self.coords[self.sample_idx]
        self.sample_idx = (self.sample_idx + 1) % self.num
        
        px, py, pz = cfg.TEST.DATA.PATCH_SIZE
        img = self.image[x:x+px, y:y+py, z:z+pz]
        msk = self.mask[x:x+px, y:y+py, z:z+pz]
        
        img = set_window_wl_ww(img, self.wl, self.ww)
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')[np.newaxis, ...]
        
        ddict = {
            'img': img,
            'gt': msk,
            'coord': np.array([[x, x+px],[y, y+py],[z, z+pz]], 'int32')
        }
        
        if self.use_heatmap:
            hp = self.heatmap[x:x+px, y:y+py, z:z+pz]
            ddict['hp'] = hp[np.newaxis, ...]

        if self.use_cascade:
            feature_map = self.feature_map[x:x + px, y:y + py, z:z + pz]
            feature_map = feature_map * 2.0 - 1.0
            feature_map = feature_map.astype('float32')[np.newaxis, ...]
            ddict['feature'] = feature_map
        
        return ddict
        
    def sample_train(self):
        
        self.sample_idx = (self.sample_idx + 1) % self.num
        
        vx, vy, vz = self.mask.shape
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE
        
        if self.random:
            x, y, z = self.sample_random_coord()
        else:
            choice = random.randint(0, 3)
            if choice <= 1:
                #x, y, z = self.points[random.randint(0, len(self.points)-1)]
                #x = x + (np.random.choice(2) * 2 - 1) * np.random.choice(8) - px // 2
                #y = y + (np.random.choice(2) * 2 - 1) * np.random.choice(8) - py // 2
                #z = z + (np.random.choice(2) * 2 - 1) * np.random.choice(8) - pz // 2
                #x = max(0, min(x, vx-px))
                #y = max(0, min(y, vy-py))
                #z = max(0, min(z, vz-pz))   #random sample from skeleton
                while 1:
                    x, y, z = self.coords[random.randint(0, len(self.coords)-1)]
                    if self.mask[x:x+px, y:y+py, z:z+pz].sum() >= 100:
                        break
                
            elif choice == 2:
                x, y, z = self.coords[random.randint(0, len(self.coords)-1)] #random sample from coords
            else:
                x, y, z = self.sample_random_coord() #random sample

        #x = random.randint(0, vx - px)        
        #y = random.randint(0, vy - py)
        #z = random.randint(0, vz - pz)
        
        #img = self.image[x:x+px, y:y+py, z:z+pz]
        #msk = self.mask[x:x+px, y:y+py, z:z+pz]
        #if msk.sum() < 10:
        #    return self.sample_train()
        
        img = self.image[x:x+px, y:y+py, z:z+pz]
        msk = self.mask[x:x+px, y:y+py, z:z+pz]
        
        img = set_window_wl_ww(img, self.wl, self.ww)
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')[np.newaxis, ...]
        
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

        if self.use_cascade:
            feature_map = self.feature_map[x:x + px, y:y + py, z:z + pz]
            feature_map = feature_map * 2.0 - 1.0
            feature_map = feature_map.astype('float32')[np.newaxis, ...]
            ddict['feature'] = feature_map

        return ddict

class VesselSubjectDS(BaseDataset):
    
    def __init__(self, para_dict, stage, VS=VesselPatchSampler):
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
        
        # if img_nii.header['pixdim'][0] > 0:
        #     seg = seg[::-1, ...]
        seg = np.transpose(seg, (1, 2, 0))
        #seg = np.transpose(seg, (0, 2, 3, 1))
        seg = seg.astype('float32' if prob else 'uint8')
        affine = img_nii.affine
        seg_img = nib.Nifti1Image(seg, affine)
        nib.save(seg_img, res_pth)    