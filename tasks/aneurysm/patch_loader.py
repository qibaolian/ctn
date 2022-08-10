import torch
from torch.utils import data
from torchvision import transforms
import os
import numpy as np
import random
import nibabel as nib
import SimpleITK as sitk

from scipy import misc
from rotation_3d import rotation3d_itk
import time

ANEURYSM_DATASET = '/home/HDD1/pancw/work/dataset/aneurysm/'

def load_image(filepath):
    img = misc.imread(filepath)
    img_shape = img.shape[1]
    img_3d = img.reshape((img_shape, img_shape, img_shape))
    return img_3d

def get_patch_coords(patch_xyz, volume_xyz, fusion=True):
    coords = []
    patch_x, patch_y, patch_z = patch_xyz[0], patch_xyz[1], patch_xyz[2]
    volume_x, volume_y, volume_z = volume_xyz[0], volume_xyz[1], volume_xyz[2]
    x = 0
    while x < volume_x:
        y = 0
        while y < volume_y:
            z = 0
            while z < volume_z:
                coords.append(
                                (x if x + patch_x < volume_x else volume_x - patch_x,
                                 y if y + patch_y < volume_y else volume_y - patch_y,
                                 z if z + patch_z < volume_z else volume_z - patch_z)
                              )
                if z + patch_z >= volume_z:
                    break
                z += patch_z // 4 if fusion else 1
            if y + patch_y >= volume_y:
                break
            y += patch_y // 4 if fusion else 1
        if x + patch_x >= volume_x:
            break
        x += patch_x // 4 if fusion else 1    
    
    return coords


class Patch3DLoader(data.Dataset):
    
    def __init__(self, train_list, train=True):
        self.patch_size = [96, 96, 96]
        self.patch_offset = [4, 4, 4]
        
        self.flip = False
        self.list = []
        self.train = train
        with open(train_list, 'r') as f:
            lines = f.readlines()
            self.list = [line.strip() for line in lines]
            if not train:
                self.list = filter(lambda x: 'positive' in x, self.list)
                          
            
    def __getitem__(self, index):
        label = 1 if 'positive' in self.list[index] else 0
        if label == 0:
            img = load_image(ANEURYSM_DATASET + self.list[index])
            mask = np.zeros(img.shape, 'uint8')
        else:
            img = load_image(ANEURYSM_DATASET + self.list[index])
            mask = load_image(ANEURYSM_DATASET + self.list[index].replace('image','aneurysm'))
            mask[mask > 128] = 1
            
        if self.train:
            if label == 1:
                img, mask = self.__positive_aug(img, mask, index)
            else:
                img, mask = self.__negative_aug(img, mask)
        else:
            img, mask = self.__center_crop(img, mask, index)
        
        img = (img / 255)*0.5 - 0.5
        
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)
        
        gt = torch.from_numpy(mask)
        return img, label, gt
    
    def __positive_aug(self, img, mask, index):
        aa = self.list[index].split('.')[0].split('_')[-6:]
        aa = [int(a) for a in aa]
        bbox = [aa[4], aa[5], aa[0], aa[1], aa[2], aa[3]]
        
        x0 = max(bbox[1]+self.patch_offset[0]-self.patch_size[0], 0)
        x1 = min(bbox[0]-self.patch_offset[0], img.shape[0]-self.patch_size[0])
        y0 = max(bbox[3]+self.patch_offset[1]-self.patch_size[1], 0)
        y1 = min(bbox[2]-self.patch_offset[1], img.shape[1]-self.patch_size[1])      
        z0 = max(bbox[5]+self.patch_offset[2]-self.patch_size[2], 0)
        z1 = min(bbox[4]-self.patch_offset[2], img.shape[2]-self.patch_size[2])  
        
        x = random.randint(x0, x1)
        y = random.randint(y0, y1)
        z = random.randint(z0, z1)
        
        return self.__crop(img, mask, (x,y,z))
        
    
    def __crop(self, img, mask, coord):
        x, y, z = coord
        img = img[x:x+self.patch_size[0],
                  y:y+self.patch_size[1],
                  z:z+self.patch_size[2]]
        mask = mask[x:x+self.patch_size[0],
                    y:y+self.patch_size[1],
                    z:z+self.patch_size[2]]
        
        if self.train and self.flip:
            flip_x = np.random.choice(2)*2 - 1
            flip_y = np.random.choice(2)*2 - 1
            flip_z = np.random.choice(2)*2 - 1
            img = img[::flip_x, ::flip_y, ::flip_z]
            mask = mask[::flip_x, ::flip_y, ::flip_z]
        
        return img.astype('float32'), mask.astype('uint8')  
    
    def __center_crop(self, img, mask, index):
        xx, yy, zz = img.shape
        if 'positive' in self.list[index]:
            aa = self.list[index].split('.')[0].split('_')[-6:]
            aa = [int(a) for a in aa]
            bbox = [aa[4], aa[5], aa[0], aa[1], aa[2], aa[3]]
            x = (bbox[0] + bbox[1]) // 2
            y = (bbox[2] + bbox[3]) // 2
            z = (bbox[4] + bbox[5]) // 2
            x = min(max(0, x - self.patch_size[0]//2), xx - self.patch_size[0])
            y = min(max(0, y - self.patch_size[1]//2), yy - self.patch_size[1])
            z = min(max(0, z - self.patch_size[2]//2), zz - self.patch_size[2])
            return self.__crop(img, mask, (x, y, z))
            
        else:
            #return self.__crop(img, mask, (0, 0, 0))
            return self.__crop(img, mask, 
                               (xx//2 - self.patch_size[0]//2, 
                                yy//2 - self.patch_size[1]//2,
                                zz//2 - self.patch_size[2]//2))
    
    def __negative_aug(self, img, mask):
        
        '''
        R_x = (np.random.choice(2) * 2 - 1)*np.random.choice(10)
        R_y = (np.random.choice(2) * 2 - 1)*np.random.choice(10)
        R_z = (np.random.choice(2) * 2 - 1)*np.random.choice(10)
        img = rotation3d_itk(img, R_x, R_y, R_z)
        mask = rotation3d_itk(mask, R_x, R_y, R_z, interpolator=sitk.sitkNearestNeighbor)
        img = np.array(img)
        mask = np.array(mask)
        
        x = random.randint(8, img.shape[0]-8-self.patch_size[0])
        y = random.randint(8, img.shape[1]-8-self.patch_size[1])
        z = random.randint(8, img.shape[2]-8-self.patch_size[2])
        '''
        
        x = random.randint(self.patch_offset[0], img.shape[0]-self.patch_offset[0]-self.patch_size[0])
        y = random.randint(self.patch_offset[1], img.shape[1]-self.patch_offset[1]-self.patch_size[1])
        z = random.randint(self.patch_offset[2], img.shape[2]-self.patch_offset[2]-self.patch_size[2])
        
        return self.__crop(img, mask, (x,y,z))
        
    def __len__(self):
        return len(self.list)

class SubjectPatch3DLoader(data.Dataset):
    
    NII_FOLDER = '/data3/pancw/data/nii_file/'
    BLOOD_FOLDER = '/data3/pancw/data/blood_mask/'
    ANEURYSM_FOLDER = '/data3/pancw/data/aneurysm_mask'

    def __init__(self, subject, overlap=True):
        t0 = time.time()
        
        self.patch_xyz = [96, 96, 96]
        self.overlap = overlap
        wl, ww = 225, 450
        w_min, w_max = wl - ww//2, wl + ww//2
        
        self.image = nib.load(os.path.join(self.NII_FOLDER, '%s.nii.gz' % subject)).get_data()
        self.image[self.image < w_min] = w_min
        self.image[self.image > w_max] = w_max
        self.image = ((1.0*(self.image - w_min) / (w_max - w_min)) * 255).astype(np.uint8)
        self.image = (self.image.astype('float32') / 255) * 0.5 - 0.5
        self.mask = nib.load(os.path.join(self.ANEURYSM_FOLDER, '%s_aneurysm.nii.gz' % subject)).get_data()
        self.mask[self.mask > 128] = 1
        
        self.blood = nib.load(os.path.join(self.BLOOD_FOLDER, '%s_blood.nii.gz' % subject)).get_data()
        self.blood[self.blood > 128] = 1
        
        coords = get_patch_coords(self.patch_xyz, self.image.shape, overlap)
        self.coords = []
        for coord in coords:
            x, y, z = coord
            bm = self.blood[x:x+self.patch_xyz[0], y:y+self.patch_xyz[1], z:z+self.patch_xyz[2]]
            #if bm[8:48, 8:48, 8:48].sum() >= 100:
            if bm.sum() > 200:
                self.coords.append(coord)
            
        np.random.shuffle(self.coords)
        

        
        print 'dataset init: %.4f' % (time.time() - t0)
     
    def __getitem__(self, index):
        #t0 = time.time()
        x, y, z = self.coords[index][0], self.coords[index][1], self.coords[index][2]
        img = self.image[x:x+self.patch_xyz[0], y:y+self.patch_xyz[1], z:z+self.patch_xyz[2]]
        img = np.transpose(img, (2, 0, 1))
        
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)       
        
        coord = np.array([[self.coords[index][0], self.coords[index][0] + self.patch_xyz[0]],
                          [self.coords[index][1], self.coords[index][1] + self.patch_xyz[1]],
                          [self.coords[index][2], self.coords[index][2] + self.patch_xyz[2]]
                         ])
        coord = torch.from_numpy(coord)
        #print 'get item: %.4f' % (time.time() - t0)
        return img, coord
    
    def volume_size(self):
        return self.image.shape
    
    def patch_size(self):
        return self.patch_xyz
    
    def mask(self, index):
        x, y, z = self.coords[index][0], self.coords[index][1], self.coords[index][2]
        return self.mask[x:x+self.patch_xyz[0], y:y+self.patch_xyz[1], z:z+self.patch_xyz[2]]
    
    def img_tensor(self, x, y, z):
        
        img = self.image[x:x+self.patch_xyz[0], y:y+self.patch_xyz[1], z:z+self.patch_xyz[2]]
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, np.newaxis, :, :, :]
        img = torch.from_numpy(img)
        
        return img
    
    def coord(self, index):
        return self.coords[index][0], self.coords[index][1], self.coords[index][2]
        
    
    def __len__(self):
        return len(self.coords)