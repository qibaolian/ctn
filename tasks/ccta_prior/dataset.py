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
        if stage == 'train':
            # data_source = np.random.choice([0, 1], p=[0.25, 0.75])
            img_path = os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '%s.nii.gz' % subject)
            # if data_source == 0:
            #     img_path = img_path.replace('vessel_largest', 'vessel_seg_largest')
            img_nii = nib.load(img_path)
        else:
            img_nii = nib.load(
                os.path.join(cfg.TEST.DATA.NII_FOLDER, '%s.nii.gz' % subject)
            )

        spacing = img_nii.header['pixdim'][1:4]
        image = img_nii.get_data()
        mask_path = os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER, '%s.nii.gz' % subject)
        if os.path.exists(mask_path):
            mask = nib.load(mask_path).get_data()
        else:
            print('{} not exists.'.format(mask_path))
            mask = np.zeros_like(image)

        ##############load skeleton and erase image ####################
        if stage == 'train':
            points = scio.loadmat(
                os.path.join(cfg.TRAIN.DATA.SKELETON_FOLDER, '%s.mat' % subject)
            )
            self.points = points['points']
            # if data_source == 0:
            #     image[image == 3] = 0
            #     image[image > 0] = 1
            # else:
            mask_erase = self.rand_erase(mask, self.points)
        else:
            self.points = []
            erase_path = os.path.join(cfg.TEST.DATA.ERASE_FOLDER, '%s.nii.gz' % subject)
            if not os.path.exists(erase_path):
                erase_path = os.path.join(cfg.TEST.DATA.ERASE_FOLDER, '%s_seg.nii.gz' % subject)
            erase_nii = nib.load(erase_path)
            mask_erase = erase_nii.get_data()
            mask_erase = mask_erase.astype('bool').astype('uint8')

        self.wl, self.ww = cfg.TRAIN.DATA.WL_WW
        image = set_window_wl_ww(image, self.wl, self.ww)
        image = image * 2.0 - 1.0
        mask = mask.astype('bool').astype('uint8')

        self.image = np.transpose(image, (2, 0, 1))
        self.mask = np.transpose(mask, (2, 0, 1))
        self.mask_erase = np.transpose(mask_erase, (2, 0, 1))
        self.spacing = np.roll(spacing, 1, axis=0)
        if len(self.points) > 0:
            self.points = np.roll(self.points, 1, axis=1)
            self.skeleton = np.zeros_like(self.mask)

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

        #########rotation################################
        if self.use_rotation and stage == 'train':
            R_x = (np.random.choice(2) * 2 - 1) * np.random.choice(15)
            R_y = (np.random.choice(2) * 2 - 1) * np.random.choice(15)
            R_z = (np.random.choice(2) * 2 - 1) * np.random.choice(15)
            tt = getTransform(self.mask.shape, R_x, R_y, R_z, (1.0, 1.0, 1.0))
            self.image = transform_image(tt, self.image, 'linear')
            self.mask = transform_image(tt, self.mask)
            self.mask_erase = transform_image(tt, self.mask_erase)
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
            # if self.use_skeleton:
            #     self.skeleton[self.mask_ignore == 1] = 255
            self.num = sample_num
            self.coords = get_patch_coords(cfg.TRAIN.DATA.PATCH_SIZE, self.mask.shape, 2)
        else:
            self.coords = get_patch_coords(cfg.TEST.DATA.PATCH_SIZE, self.mask.shape, 2)
            self.num = len(self.coords)
 
        self.sample_idx = 0
    
    def __len__(self):
        return self.num
    
    def rand_erase(self, mask, points):
        '''
        erase the input mask to calculate loss
        :param mask: vessel mask
        :param points: skeleton
        :return: mask_erase
        '''
        mask_ske = np.zeros_like(mask)
        mask_ske[points[:, 0], points[:, 1], points[:, 2]] = 1
        np.random.shuffle(points)
        end_points = self.FindEndPoints(points, mask_ske.astype(bool))
        np.random.shuffle(end_points)

        radius_thres = 10
        erased_num_mid = 5
        erased_num_end = 0
        mask_erase = mask.copy()

        r = 12
        n_point = 0
        for point in points:
            x, y, z = point
            point_radius = self.cal_radius3d(mask, point)
            # skip large, main vessel
            sub_mask = mask_erase[x - r:x + r, y - r:y + r, z - r:z + r]
            if point_radius > radius_thres or (sub_mask == 1).sum() > 0:
                continue
            mask_erase[x - r:x + r, y - r:y + r, z - r:z + r] = (sub_mask > 0) * 4
            n_point += 1
            if n_point == erased_num_mid:
                break

        r = 6
        n_point = 0
        for point in end_points:
            x, y, z = point
            # skip main vessel
            sub_mask = mask_erase[x - r:x + r, y - r:y + r, z - r:z + r]
            if (sub_mask == 1).sum() > 0:
                continue
            # skip mid erase
            if (sub_mask == 4).sum() > 0:
                continue
            mask_erase[x - r:x + r, y - r:y + r, z - r:z + r] = (sub_mask > 0) * 5
            n_point += 1
            if n_point == erased_num_end:
                break

        mask_erase[mask_erase == 4] = 0
        mask_erase[mask_erase == 5] = 0
        mask_erase[mask_erase > 0] = 1
        return mask_erase

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
        # img = set_window_wl_ww(img, self.wl, self.ww)
        # img = img * 2.0 - 1.0
        img = img.astype('float32')[np.newaxis, ...]
        msk = self.mask[x:x+px, y:y+py, z:z+pz]
        mask_erase = self.mask_erase[x:x + px, y:y + py, z:z + pz]
        mask_erase = mask_erase.astype('float32')[np.newaxis, ...]

        ddict = {
            'img': img,
            'gt': msk,
            'coord': np.array([[x, x+px],[y, y+py],[z, z+pz]], 'int32')
        }
        ddict['erase'] = mask_erase

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
        flip_y = np.random.choice(2) * 2 - 1
        flip_z = np.random.choice(2) * 2 - 1
        img = self.image[x:x+px, y:y+py, z:z+pz]
        # img = set_window_wl_ww(img, self.wl, self.ww)
        # img = img * 2.0 - 1.0
        img = img.astype('float32')[np.newaxis, ...]
        img = img[..., ::flip_y, ::flip_z].copy()
        msk = self.mask[x:x+px, y:y+py, z:z+pz]
        msk = msk[..., ::flip_y, ::flip_z].copy()
        mask_erase = self.mask_erase[x:x + px, y:y + py, z:z + pz]
        mask_erase = mask_erase.astype('float32')[np.newaxis, ...]
        mask_erase = mask_erase[..., ::flip_y, ::flip_z].copy()

        ddict = {'img': img, 'gt': msk, 'subject': self.subject, 'coord': np.array([[x, x+px],[y, y+py],[z, z+pz]], 'int32')}
        ddict['erase'] = mask_erase
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

    def cal_radius3d(self, mask, point):
        width, height, depth = mask.shape
        x, y, z = point

        # along x
        coord = x
        while (coord > 0 and mask[coord, y, z] > 0):
            coord -= 1
        x_min = coord
        coord = x
        while (coord < width and mask[coord, y, z] > 0):
            coord += 1
        x_max = coord

        # along y
        coord = y
        while (coord > 0 and mask[x, coord, z] > 0):
            coord -= 1
        y_min = coord
        coord = y
        while (coord < height and mask[x, coord, z] > 0):
            coord += 1
        y_max = coord

        # along z
        coord = z
        while (coord > 0 and mask[x, y, coord] > 0):
            coord -= 1
        z_min = coord
        coord = z
        while (coord < depth and mask[x, y, coord] > 0):
            coord += 1
        z_max = coord
        radius3d = min([x_max - x_min, y_max - y_min, z_max - z_min])
        return radius3d

    def FindEndPoints(self, points, volume):
        '''
            在中心线上寻找端点
        '''

        pts = filter(lambda x: np.sum(volume[x[0] - 1:x[0] + 2,
                                      x[1] - 1:x[1] + 2,
                                      x[2] - 1:x[2] + 2]) <= 2, points)
        return np.array(list(pts))

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
            self.cache_samplers.append(
                self.VS(
                    subject,
                    self.sample_num_per_subject,
                    stage='train'
                )
            )
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
        self.data_sampler = self.VS(
            self.subject,
            -1,
            stage='validate'
        )
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