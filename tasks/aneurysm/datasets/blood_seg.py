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
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

class VesselPatchSampler(BaseSubjectSampler):
    
    def __init__(self, subject):
        
        super(VesselPatchSampler, self).__init__(subject)
        img_nii = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '%s.nii.gz' % subject))
        image = img_nii.get_data()
        mask = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER, '%s.nii.gz' % subject)).get_data()
        self.image, self.mask = np.transpose(image, (2, 0, 1)), np.transpose(mask, (2, 0, 1))        
        self.mask = self.mask.astype('bool').astype('uint8')      
        self.has_sdm = 'SDM_FOLDER' in cfg.TRAIN.DATA.keys()
        if self.has_sdm:
            sdm = nib.load(os.path.join(cfg.TRAIN.DATA.SDM_FOLDER, '%s.nii.gz' % subject)).get_data()
            self.sdm = np.transpose(sdm, (2, 0, 1))
        
        self.spacing = img_nii.header['pixdim'][1:4]
        self.spacing = np.roll(self.spacing, 1, axis=0)
        #normalize_spacing
        if self.spacing[0] > 0.625:
            #self.image = self.normalize_spacing(self.image, self.spacing[0]/0.625, False)
            #self.mask  = self.normalize_spacing(self.mask,  self.spacing[0]/0.625)
            x, y, z = self.image.shape
            shape = (int(x*self.spacing[0]/0.625), y, z)
            self.image = self.tensor_resize(self.image,  shape, False)
            self.mask = self.tensor_resize(self.mask, shape, True)
            if self.has_sdm:
                self.sdm = self.tensor_resize(self.sdm, shape, False)
            
        if cfg.TRAIN.DATA.POSITION == 'Head':
            num = self.image.shape[0] // 3 + 64
            self.image, self.mask = self.image[-num:, ...], self.mask[-num:, ...]
            if self.has_sdm:
                self.sdm = self.sdm[-num:, ...]
                           
        elif cfg.TRAIN.DATA.POSITION == 'Neck':
            num = self.image.shape[0] // 3
            self.image, self.mask = self.image[0:num, ...], self.mask[0:num, ...] 
            if self.has_sdm:
                self.sdm = self.sdm[0:num, ...]

        if 'SKELETON_FOLDER' in cfg.TRAIN.DATA.keys():
            points = scio.loadmat(os.path.join(cfg.TRAIN.DATA.SKELETON_FOLDER, '%s.mat' % subject))['points']
            self.skeleton = np.roll(points, 1, axis=1)
        else:
            self.skeleton = []
        
        self.wl, self.ww = cfg.TRAIN.DATA.WL_WW
           
        #rotation_transform
        #self.rotation_transform()
        
        #normalize_ntensity
        if cfg.MODEL.INPUT_CHANNEL > 1:
            img1 = set_window_wl_ww(self.image, 250, 600)
            img2 = set_window_wl_ww(self.image, 300, 800)
            img2 = set_window_wl_ww(self.image, 350, 1000)
            self.image = np.concatenate((img1[np.newaxis, ...], img2[np.newaxis, ...], img2[np.newaxis, ...]))
        else:
            self.image = set_window_wl_ww(self.image, self.wl, self.ww)[np.newaxis, ...]
        #self.mean, self.var = self.image.mean(), self.image.std()
        #self.image = self.normalize_intensity(self.image)
        
    @staticmethod
    def tensor_resize(image, shape, is_label=True):
        x = torch.from_numpy(image[np.newaxis, np.newaxis, ...])
        if is_label:
            y = F.interpolate(x.float(), shape, mode='nearest')
        else:
            y = F.interpolate(x.float(), shape, mode= 'trilinear', align_corners=True)
        y = y.numpy().astype(image.dtype)[0, 0]
        return y
        
    @staticmethod
    def normalize_spacing(image, factor, is_label=True):
        order = 0 if is_label == True else 3        
        return interpolation.zoom(image, (factor, 1.0, 1.0), order=order, mode='nearest')
        
    @staticmethod
    def normalize_intensity(image, mean_std=True):
        if mean_std:
            mean, std = image.mean(), image.std()
            image = (image - mean) / std
        else:
            image = (image / 255.0) * 2.0 - 1.0
            
        return image.astype('float32')
    
    @staticmethod
    def zoom(image, shape, is_label=True):
        order = 0 if is_label == True else 3
        scale = np.divide(np.array(shape, dtype=float), np.array(image.shape, dtype=float))
        return interpolation.zoom(image, scale, order=order, mode='nearest')
    
    def rotation_transform(self):
        R_x = (np.random.choice(2) * 2 - 1)*np.random.choice(10)
        R_y = (np.random.choice(2) * 2 - 1)*np.random.choice(10)
        R_z = (np.random.choice(2) * 2 - 1)*np.random.choice(10)
        
        tt = getTransform(x.shape, R_x, R_y, R_z)
        
        self.image = transform_image(tt, self.image, 'linear')
        self.mask = transform_image(tt, self.mask)

    def sample_one(self):
        
        #scale = random.uniform(0.75, 1.33)
        vx, vy, vz = self.image.shape[-3:]
        px, py, pz = cfg.TRAIN.DATA.PATCH_SIZE
        
        #npy = int(np.round(py / scale))
        #npz = int(np.round(pz / scale))
        
        npy, npz = py, pz
        
        x = random.randint(0, vx - px)        
        y = random.randint(0, vy - npy)
        z = random.randint(0, vz - npz)
        
        img = self.image[:, x:x+px, y:y+npy, z:z+npz]
        gt = self.mask[x:x+px, y:y+npy, z:z+npz]        
        if self.has_sdm:
            sdm = self.sdm[x:x+px, y:y+npy, z:z+npz]
        
        return {'img': img, 'gt': gt, 'sdm': sdm} if self. has_sdm else {'img': img, 'gt': gt}

class Patch3DLoader(data.Dataset):
    
    def __init__(self, sampler, memory):
        self.sampler = sampler
        self.flip = True
        self.memory = memory
        if self.memory:
            self.data = self.sampler.get_data()
        
    
    def __getitem__(self, index):
        
        if self.memory:
            data = self.data[index]
            img, gt = data['img'], data['gt']
        else:
            data = self.sampler.get_data()
            img, gt = data['img'], data['gt']
                
        img = (img / 255.0) * 2.0 - 1.0
        if self.flip:
            flip_y = np.random.choice(2)*2 - 1
            flip_z = np.random.choice(2)*2 - 1
            img = img[:, :, ::flip_y, ::flip_z]
            gt = gt[::, ::flip_y, ::flip_z]
        
        img = img.astype('float32')
        gt = gt.astype('uint8')
        img, gt = torch.from_numpy(img), torch.from_numpy(gt).long()
        
        if 'sdm' in data:
            sdm = data['sdm']
            if self.flip:
                sdm = sdm[::, ::flip_y, ::flip_z]
            sdm = torch.from_numpy(sdm.astype('float32'))
            
            return {'img': img, 'gt': gt, 'sdm': sdm}
        
        return {'img': img, 'gt': gt}
        
    def __len__(self):  
        return len(self.sampler)
    
class VesselDatasetSampler(object):
    
    def __init__(self, train_lst, memory=False):
        self.memory = memory
        with open(train_lst, 'r') as f:
            lines = f.readlines()
            self.subjects = [line.strip() for line in lines]
        self.sampler = None
        #for subject in self.subjects:
        #    ss = VesselPatchSampler(subject)#self.subjects[0])
        #    dd = ss.sample_one()
        
    def asyn_sample(self, subject_num=100, patch_num=100):
        
        subject_num = min(subject_num, len(self.subjects))
        samples = np.random.choice(range(len(self.subjects)), size=subject_num, replace= False)
        subjects = [self.subjects[idx] for idx in samples]
        
        if self.memory:
            self.sampler = MemoryDatasetSampler(subjects, patch_num, VesselPatchSampler)
        else:
            self.sampler = FileDatasetSampler(subjects, patch_num, VesselPatchSampler)
    
    def get_data_loader(self):        
        return Patch3DLoader(self.sampler, self.memory)

class BLOOD_SEG(BaseDataset):
    def __init__(self, para_dict, stage):
        super(BLOOD_SEG, self).__init__(para_dict, stage)

    def train_init(self):
        self.train_list = self.para_dict.get("train_list", None)
        with open(self.train_list, 'r') as f:
            lines = f.readlines()
            self.npz_files = [line.strip() for line in lines]
        random.shuffle(self.npz_files)
        self.num = len(self.npz_files)
        
        self.rotate_transform = False
        self.use_bbox = True
        self.flip = True

    def val_init(self):
        
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
            self.msk = self.msk.astype('bool').astype('uint8')
            self.spacing = np.roll(self.spacing, 1, axis=0)
            self.origin_shape = self.img.shape
            
            if self.spacing[0] > 0.625:
                x, y, z = self.origin_shape
                shape = (int(x*self.spacing[0]/0.625), y, z)
                self.img = VesselPatchSampler.tensor_resize(self.img,  shape, False)
                self.msk = VesselPatchSampler.tensor_resize(self.msk, shape, True)
            
            if cfg.TRAIN.DATA.POSITION == 'Head':  
                num = self.img.shape[0] // 3 + 64
                self.img, self.msk = self.img[-num:, ...], self.msk[-num:, ...]        
            elif cfg.TRAIN.DATA.POSITION == 'Neck':
                num = self.img.shape[0] // 3
                self.img, self.msk = self.img[0:-num, ...], self.msk[0:-num, ...] 
            
            WL, WW = cfg.TRAIN.DATA.WL_WW
            
            if cfg.MODEL.INPUT_CHANNEL > 1:
                img1 = set_window_wl_ww(self.img, 250, 600)
                img2 = set_window_wl_ww(self.img, 300, 800)
                img2 = set_window_wl_ww(self.img, 350, 1000)
                self.img = np.concatenate((img1[np.newaxis, ...], img2[np.newaxis, ...], img2[np.newaxis, ...]))
            else:
                self.img = set_window_wl_ww(self.img, WL, WW)[np.newaxis, ...]
            
            self.patch_size = cfg.TEST.DATA.PATCH_SIZE
            self.coords = get_patch_coords(self.patch_size, self.img.shape[-3:], 2)
            self.num = len(self.coords)            
            self.use_volume = True
            
        else:

            self.val_list = self.para_dict.get("val_list", None)
            self.on_line = '.lst' not in self.val_list
            if self.on_line: 
                with h5py.File(self.val_list, 'r') as f:
                    self.img_lst, self.gt_lst = f['img'][:], f['gt'][:]
                    self.num = len(self.img_lst)
            else:
                with open(self.val_list, 'r') as f:
                    lines = f.readlines()
                    self.npz_files = [line.strip() for line in lines]
                    self.num = len(self.npz_files)
                
    def test_init(self):
        self.subject = self.para_dict.get("subject", None)
        img_nii = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '%s.nii.gz' % self.subject))
        self.spacing = img_nii.header['pixdim'][1:4]
        img = img_nii.get_data()
    
        self.img = np.transpose(img, (2, 0, 1))
        self.spacing = np.roll(self.spacing, 1, axis=0)
        if self.spacing[0] > 0.625:
            #self.img = VesselPatchSampler.normalize_spacing(self.img, self.spacing[0]/0.625, False)
            x, y, z = self.img.shape
            shape = (int(x*self.spacing[0]/0.625), y, z)
            self.img = VesselPatchSampler.tensor_resize(self.img,  shape, False)    
        
        if cfg.TEST.DATA.POSITION == 'Head':
            num = self.img.shape[0] // 3 + 64
            self.img = self.img[-num:, ...]
        elif cfg.TEST.DATA.POSITION == 'Neck':
            num = self.img.shape[0] // 3
            self.img = self.img[0:-num, ...]
        
        if self.spacing[0] > 0.625:
            self.origin_shape = (int(self.img.shape[0]*0.625/self.spacing[0]), self.img.shape[1], self.img.shape[2])
        else:
            self.origin_shape = self.img.shape
        
        WL, WW = cfg.TRAIN.DATA.WL_WW
        #self.img = set_window_wl_ww(self.img, WL, WW)
        if cfg.MODEL.INPUT_CHANNEL > 1:
            img1 = set_window_wl_ww(self.img, 250, 600)
            img2 = set_window_wl_ww(self.img, 300, 800)
            img2 = set_window_wl_ww(self.img, 350, 1000)
            self.img = np.concatenate((img1[np.newaxis, ...], img2[np.newaxis, ...], img2[np.newaxis, ...]))
        else:
            self.img = set_window_wl_ww(self.img, WL, WW)[np.newaxis, ...]
        
        self.patch_size =  cfg.TEST.DATA.PATCH_SIZE
        self.coords = get_patch_coords(self.patch_size, self.img.shape[-3:], 2)
        self.num = len(self.coords)
    
    def rotate_transformation(self, x, R, interp='linear'):        
        tt = getTransform(x.shape, R[0], R[1], R[2])
        if interp == 'linear':
            x = transform_image(tt, x, 'linear')
        else:
            x = transform_image(tt, x)
        return x
    
    def train_load(self, index):        
        #npz_pth = self.npz_files[index]
        #npz = np.load(npz_pth)
        #img, gt, hp, bbox = npz['img'], npz['gt'], npz['heatmap'], npz['bbox']
        #img, gt, bbox = npz['img'], npz['gt'], npz['bbox']
        #npz.close()
        with h5py.File(self.npz_files[index], 'r') as f:
            img, gt, bbox = f['img'][:], f['gt'][:], f['bbox'][:]
        
        z1, x1, y1 = img.shape
        z0, x0, y0 = cfg.TRAIN.DATA.PATCH_SIZE
        z = random.randint(0, z1 - z0)
        
        #img, gt, hp, bbox= img[z:z+z0], gt[z:z+z0], hp[z:z+z0], bbox[z:z+z0]
        img, gt, bbox= img[z:z+z0], gt[z:z+z0], bbox[z:z+z0]
        
        ###################################sample patch#################################
        if self.use_bbox:
            pad = 10
            xmin, ymin = max(min([b[0] for b in bbox])-pad, 0), max(min([b[1] for b in bbox])-pad, 0)
            xmax, ymax = min(max([b[0]+b[2] for b in bbox])+pad, x1), min(max([b[1]+b[3] for b in bbox])+pad, y1)
            ww, hh = xmax - xmin, ymax - ymin
            xmin = xmin - max(0, x0 - ww) // 2
            ymin = ymin - max(0, y0 - hh) // 2        
            xmin = max(0, min(xmin, x1 - x0))
            ymin = max(0, min(ymin, y1 - y0))
            x = xmin + random.randint(0, max(x0, ww) - x0)
            y = ymin + random.randint(0, max(y0, hh) - y0)
        else:
            x, y = random.randint(0, x1 - x0), random.randint(0, y1 - y0)
        ###################################sample patch#################################
        
        #####################################rotation###################################
        if self.rotate_transform:
            R_x = (np.random.choice(2) * 2 - 1)*np.random.choice(30)
            img = self.rotate_transformation(img, (R_x, 0, 0))
            gt = self.rotate_transformation(gt, (R_x, 0, 0), 'nearest')
            #hp = self.rotate_transformation(hp, (R_x, 0, 0), 'nearest')
       #####################################rotation###################################
    
        #img, gt, hp = img[:, x:x+x0, y:y+y0], gt[:, x:x+x0, y:y+y0], hp[:, x:x+x0, y:y+y0]
        img, gt = img[:, x:x+x0, y:y+y0], gt[:, x:x+x0, y:y+y0]
        ################################flip###########################################
        if self.flip:
            f_x, f_y = np.random.choice(2)*2 - 1, np.random.choice(2)*2 - 1
            #img, gt, hp = img[:, ::f_x, ::f_y].copy(), gt[:, ::f_x, ::f_y].copy(), hp[:, ::f_x, ::f_y].copy()
            img, gt = img[:, ::f_x, ::f_y].copy(), gt[:, ::f_x, ::f_y].copy()
        ################################flip##########################################

        #hp = hp.astype('float32')
        gt = gt.astype('uint8')
        #hp = hp[np.newaxis, :, :, :]
        WL, WW = cfg.TRAIN.DATA.WL_WW
        img = set_window_wl_ww(img, WL, WW)
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)

        gt = torch.from_numpy(gt)
        #hp = torch.from_numpy(hp)

        return {'img': img, 'gt': gt}#, 'heatmap': hp}

    def test_load(self, index):
        x, y, z = self.coords[index]

        img = self.img[:, x:x + self.patch_size[0],
                               y:y + self.patch_size[1],
                               z:z + self.patch_size[2]]

        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')
        img = torch.from_numpy(img)

        coord = np.array([[x, x + self.patch_size[0]],
                          [y, y + self.patch_size[1]],
                          [z, z + self.patch_size[2]]])
        coord = torch.from_numpy(coord)

        return img, coord

    def val_load(self, index): 
        
        if self.use_volume:
            x, y, z = self.coords[index]
            img = self.img[:, x:x + self.patch_size[0],
                                   y:y + self.patch_size[1],
                                   z:z + self.patch_size[2]]
            
            msk = self.msk[x:x + self.patch_size[0],
                                 y:y + self.patch_size[1],
                                 z:z + self.patch_size[2]]
            
            img = (img / 255.0) * 2.0 - 1.0
            img = img.astype('float32')
            img = torch.from_numpy(img)
            msk = torch.from_numpy(msk)
            
            coord = np.array([[x, x + self.patch_size[0]],
                                     [y, y + self.patch_size[1]],
                                     [z, z + self.patch_size[2]]])
            coord = torch.from_numpy(coord) 
            return img, msk, coord
        
        else:
            if self.on_line:
                image, mask = self.img_lst[index], self.gt_lst[index]
            else:
                with h5py.File(self.npz_files[index], 'r') as f:
                    image, mask = f['img'][:], f['gt'][:]

            image = (image / 255.0) * 2.0 - 1.0
            image = image.astype('float32')
            image = image[np.newaxis, ...]
            mask = mask.astype('uint8')
            #heatmap = heatmap.astype('float32')
            #heatmap = heatmap[np.newaxis, ...]
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            #heatmap = torch.from_numpy(heatmap)        
            return {'img': image, 'gt': mask} #, 'heatmap': heatmap}

    def volume_size(self):
        return self.img.shape[-3:]
    
    def volume_mask(self):        
        return self.msk
    
    def save(self, seg, res_pth, prob=False):
        img = nib.load(os.path.join(cfg.TEST.DATA.NII_FOLDER, '%s.nii.gz' % self.subject))
        seg = seg.astype('float32' if prob else 'uint8')
        if self.spacing[0] > 0.625:
            #seg = VesselPatchSampler.normalize_spacing(seg, 0.625/self.spacing[0], False if prob else True)
            seg = VesselPatchSampler.tensor_resize(seg, self.origin_shape, False if prob else True)
        
        seg = np.transpose(seg, (1, 2, 0))
        
        msk = np.zeros(img.header['dim'][1:4], 'float32' if prob else 'uint8')
        if cfg.TEST.DATA.POSITION == 'Head':
            msk[:, :, -seg.shape[2]:] = seg
        elif cfg.TEST.DATA.POSITION == 'Neck':
            msk[:,:, 0:seg.shape[2]] = seg
        else:
            msk = seg
            
        seg_nii = nib.Nifti1Image(msk, img.affine)
        nib.save(seg_nii, res_pth)


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
        # for compat python v2.*.*
        # self.next = self.__next__
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preload(self):
        data = next(self.loader)  # may rasie StopIteration
        
        self.data = {}
        with torch.cuda.stream(self.stream):
            for k, v in data.items():
                self.data[k] = v.cuda(non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        self.preload()
        return self.data

    def __len__(self):
        return len(self.loader)