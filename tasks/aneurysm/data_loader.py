import torch
from torch.utils import data
from torchvision import transforms
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from scipy import misc
from rotation_3d import rotation3d_itk
import time



positive_image_folder = '/data3/pancw/data/patch/dataset/train/image'
aneurysm_mask_folder = '/data3/pancw/data/patch/dataset/train/aneurysm'
blood_mask_folder = '/data3/pancw/data/patch/dataset/train/blood'
negative_image_folder = '/data3/pancw/data/patch/dataset/train/health'


def load_image(filepath):
    img = misc.imread(filepath)
    img_shape = img.shape[1]
    img_3d = img.reshape((img_shape, img_shape, img_shape))
    return img_3d

def get_patch_coords(patch_xyz, volume_xyz, fusion=4):
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
                z += patch_z // fusion
            if y + patch_y >= volume_y:
                break
            y += patch_y // fusion
        if x + patch_x >= volume_x:
            break
        x += patch_x // fusion    
    
    return coords

class Patch3DLoader(data.Dataset):
    
    def __init__(self, train_list, train=True, rule='all'):
        self.crop = [64, 64, 64]
        self.flip = False
        self.list = []
        self.train = train
        with open(train_list, 'r') as f:
            lines = f.readlines()
            self.list = [line.strip() for line in lines]
            if rule == 'aneurysm':
                self.list = filter(lambda x: 'aneurysm' in x, self.list)
            else:
                self.list = filter(lambda x: 'negative' not in x, self.list)
        
        if not self.train:
            self.shift = []
            self.offset = []
            for i in range(len(self.list)):
                shift_x = 0#(np.random.choice(2) * 2 - 1)*np.random.choice(8) 
                shift_y = 0#(np.random.choice(2) * 2 - 1)*np.random.choice(8)
                shift_z = 0#(np.random.choice(2) * 2 - 1)*np.random.choice(8)

                offset_w = 0#(np.random.choice(2) * 2 - 1)*np.random.choice(4) 
                offset_h = 0#(np.random.choice(2) * 2 - 1)*np.random.choice(4)
                offset_d = 0#(np.random.choice(2) * 2 - 1)*np.random.choice(4)
                
                self.shift.append([shift_x, shift_y, shift_z])
                self.offset.append([offset_w, offset_h, offset_d])
                          
            
    def __getitem__(self, index):
        label = 1 if 'aneurysm' in self.list[index] else 0
        if label == 0:
            img = load_image(self.list[index])
            mask = np.zeros(img.shape, 'uint8')
        else:
            img = load_image(self.list[index])
            mask = load_image(os.path.join(aneurysm_mask_folder, self.list[index].split('/')[-1]))
            mask[mask > 128] = 1
            num = mask.sum()
            
        img, mask = self.__data_aug(img, mask, index)
        if label == 1:
            if 1.0 * mask.sum() / num < 0.5:
                label = 0
        
        
        img = (img / 255)*0.5 - 0.5
        #img = (img - np.mean(img)) / (np.std(img) + 1e-4)
        
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)
        
        y = mask
        
        gt = torch.from_numpy(y)
        return img, label, gt
        
    def __data_aug(self, img, mask, index=0):
        
        R_x = (np.random.choice(2) * 2 - 1)*np.random.choice(15)
        R_y = (np.random.choice(2) * 2 - 1)*np.random.choice(15)
        R_z = (np.random.choice(2) * 2 - 1)*np.random.choice(15)
        
        #zoom = [1.0, 1.0, 1.0]
        #shear = [0, 0, 0, 0, 0, 0]
        if self.train:
            img = rotation3d_itk(img, R_x, R_y, R_z)
            mask = rotation3d_itk(mask, R_x, R_y, R_z, interpolator=sitk.sitkNearestNeighbor)
            img = np.array(img)
            mask = np.array(mask)
        
        if self.train:
            shift_x = (np.random.choice(2) * 2 - 1)*np.random.choice(4) 
            shift_y = (np.random.choice(2) * 2 - 1)*np.random.choice(4)
            shift_z = (np.random.choice(2) * 2 - 1)*np.random.choice(4)

            offset_w = (np.random.choice(2) * 2 - 1)*np.random.choice(4) 
            offset_h = (np.random.choice(2) * 2 - 1)*np.random.choice(4)
            offset_d = (np.random.choice(2) * 2 - 1)*np.random.choice(4)
            
            crop_x = np.shape(img)[0]//2 - self.crop[0]//2 - shift_x - offset_w
            crop_y = np.shape(img)[1]//2 - self.crop[1]//2 - shift_y - offset_h
            crop_z = np.shape(img)[2]//2 - self.crop[2]//2 - shift_z - offset_d
        else:
            shift_x, shift_y, shift_z = self.shift[index]
            offset_w, offset_h, offset_d = self.offset[index]
            crop_x = np.shape(img)[0]//2 - self.crop[0]//2 - shift_x - offset_w
            crop_y = np.shape(img)[1]//2 - self.crop[1]//2 - shift_y - offset_h
            crop_z = np.shape(img)[2]//2 - self.crop[2]//2 - shift_z - offset_d
            
        img = img[crop_x:crop_x+self.crop[0],
                  crop_y:crop_y+self.crop[1],
                  crop_z:crop_z+self.crop[2]]
        mask = mask[crop_x:crop_x+self.crop[0],
                    crop_y:crop_y+self.crop[1],
                    crop_z:crop_z+self.crop[2]]
        
        if self.flip:
            flip_x = np.random.choice(2)*2 - 1
            flip_y = np.random.choice(2)*2 - 1
            flip_z = np.random.choice(2)*2 - 1
            img = img[::flip_x, ::flip_y, ::flip_z]
            mask = mask[::flip_x, ::flip_y, ::flip_z]
        
        return img.astype('float32'), mask.astype('uint8')
        
    def __len__(self):
        return len(self.list)

class SubjectPatch3DLoader(data.Dataset):
    
    NII_FOLDER = '/data3/pancw/data/nii_file/'
    BLOOD_FOLDER = '/data3/pancw/data/blood_mask/'
    ANEURYSM_FOLDER = '/data3/pancw/data/aneurysm_mask'

    def __init__(self, subject, overlap=1):
        t0 = time.time()
        
        self.patch_xyz = [64, 64, 64]
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
        
        #self.coords = get_patch_coords(self.patch_xyz, self.image.shape, overlap)
        coords = get_patch_coords(self.patch_xyz, self.image.shape, overlap)
        self.coords = []
        for coord in coords:
            x, y, z = coord
            bm = self.blood[x:x+self.patch_xyz[0], y:y+self.patch_xyz[1], z:z+self.patch_xyz[2]]
            #if bm[8:48, 8:48, 8:48].sum() >= 100:
            if bm.sum() > 100:
                self.coords.append(coord)
            
        np.random.shuffle(self.coords)
        

        
        print('dataset init: %.4f' % (time.time() - t0))
     
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
    

class Volume3DLoader(data.Dataset):
    
    IMG_FOLDER = '/data2/pancw/data/resize/256X512X512/img'
    GT_FOLDER = '/data2/pancw/data/resize/256X512X512/gt'
    
    def __init__(self, subject_lst):
        with open(subject_lst) as f:
            lines = f.readlines()
        self.subjects = [line.strip() for line in lines]
    
    def __getitem__(self, index):
        subject = self.subjects[index]
        #t0 = time.time()
        img = np.load(os.path.join(self.IMG_FOLDER, '%s.npy' % subject))
        gt = np.load(os.path.join(self.GT_FOLDER, '%s_aneurysm.npy' % subject))
        #print 't0: ', time.time() - t0
        
        #t0 = time.time()
        #img = img.astype('float32')
        #img = (img / 255)*0.5 - 0.5
        #print 't1: ', time.time() - t0
        
        #t0 = time.time()
        #gt[gt>128] = 1
        #print 't2: ', time.time() - t0
        
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)
        
        y = gt
        
        gt = torch.from_numpy(y)
        return img, 1, gt   
    
    def __len__(self):
        return len(self.subjects)