import os
import numpy as np
import torch
from torch.utils import data
import random
import nibabel as nib
from scipy import misc
from joblib import Parallel, delayed
from tasks.aneurysm.datasets.blood_seg import BLOOD_SEG
from utils.config.defaults import _C as cfg

from .rotation_3d import getTransform, transform_image



def set_window_wl_ww(tensor, wl=225, ww=450):
    
    w_min, w_max = wl - ww//2, wl + ww//2
    tensor[tensor < w_min] = w_min
    tensor[tensor > w_max] = w_max
    tensor = ((1.0 * (tensor - w_min) / (w_max - w_min)) * 255).astype(np.uint8)
    
    return tensor

def rotation_subject(img, mask):
    
    R_x = (np.random.choice(2) * 2 - 1)*np.random.choice(10)
    #R_y = (np.random.choice(2) * 2 - 1)*np.random.choice(15)
    #R_z = (np.random.choice(2) * 2 - 1)*np.random.choice(15)
    
    R_y = R_z = 0
    
    tt = getTransform(img.shape, R_x, R_y, R_z)
    img[:] = transform_image(tt, img, 'linear')
    mask[:] = transform_image(tt, mask)
    
def load_data(Sampler, subject, patch_num=100, rotation=True):
    #t0 = time.time()
    img = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER,
                   '%s.nii.gz' % subject)).get_data()
    #blood = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER,
    #                    '%s_blood.nii.gz' % subject)).get_data()
    blood = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER,
                        '%s.nii.gz' % subject)).get_data()
    
    img = np.transpose(img, (2, 0, 1))
    WL, WW = cfg.TRAIN.DATA.WL_WW
    img = set_window_wl_ww(img, WL, WW)
    
    blood = np.transpose(blood, (2, 0, 1))
    blood[blood > 128] = 1
    #print 'data load need %f' % (time.time() - t0)
    
    if rotation:
        rotation_subject(img, blood)
    
    sampler = Sampler(blood)
    coords = sampler.sample(patch_num)
    
    return img, blood, coords

def save_image_patch(img, coord, img_name):
    patch = img[coord[0]:coord[0]+cfg.TRAIN.DATA.PATCH_SIZE[0],
                coord[1]:coord[1]+cfg.TRAIN.DATA.PATCH_SIZE[1],
                coord[2]:coord[2]+cfg.TRAIN.DATA.PATCH_SIZE[2]]
    patch = set_window_wl_ww(patch)
    misc.imsave(img_name,
                patch.reshape(cfg.TRAIN.DATA.PATCH_SIZE[0]*cfg.TRAIN.DATA.PATCH_SIZE[1], cfg.TRAIN.DATA.PATCH_SIZE[2]))

def save_blood_patch(blood, coord, img_name):
    patch = blood[coord[0]:coord[0]+cfg.TRAIN.DATA.PATCH_SIZE[0],
                  coord[1]:coord[1]+cfg.TRAIN.DATA.PATCH_SIZE[1],
                  coord[2]:coord[2]+cfg.TRAIN.DATA.PATCH_SIZE[2]]
    misc.imsave(img_name,
                patch.reshape(cfg.TRAIN.DATA.PATCH_SIZE[0]*cfg.TRAIN.DATA.PATCH_SIZE[1], cfg.TRAIN.DATA.PATCH_SIZE[2]))
    
def generae_patch(subject, save_folder):
    
    img = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER,
                   '%s.nii.gz' % subject)).get_data()
    blood = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER,
                     '%s_blood.nii.gz' % subject)).get_data()
    
    img = np.transpose(img, (2, 0, 1))
    blood = np.transpose(blood, (2, 0, 1))
    
    sampler = BloodSampler(blood)
    
    coords = []    
    k = 20    
    coords.extend(sampler.sample_traverse(k))
    coords.extend(sampler.sample_random(k))
    
    for coord in coords:
        img_name = '%s/image/%s_image_%d_%d_%d.png' % (save_folder, subject,
                                                 coord[0], coord[1], coord[2])
        save_image_patch(img, coord, img_name)
        
        img_name = '%s/mask/%s_blood_%d_%d_%d.png' % (save_folder, subject,
                                                 coord[0], coord[1], coord[2])
        save_blood_patch(blood, coord, img_name)

def generate_validation_dataset():
    
    with open('/data3/pancw/data/patch/dataset/train/test_subjects.lst', 'r') as f:
        lines = f.readlines()
        subjects = [line.strip() for line in lines]
    save_folder = '/data3/pancw/data/patch/dataset/blood_validate'
    k = len(subjects) // 2
    Parallel(n_jobs=k)(delayed(generae_patch)
                       (subject, save_folder)
                       for subject in subjects)
    
def get_patch_coords(patch_xyz, volume_xyz, stride=2):
    
    coords = []
    p_x, p_y, p_z = patch_xyz[0], patch_xyz[1], patch_xyz[2]
    v_x, v_y, v_z = volume_xyz[0], volume_xyz[1], volume_xyz[2]
    x = 0
    while x < v_x:
        y = 0
        while y < v_y:
            z = 0
            while z < v_z:
                coords.append(
                                (x if x + p_x < v_x else v_x - p_x,
                                 y if y + p_y < v_y else v_y - p_y,
                                 z if z + p_z < v_z else v_z - p_z)
                             )
                if z + p_z >= v_z:
                    break
                z += p_z // stride
            if y + p_y >= v_y:
                break
            y += p_y // stride
        if x + p_x >= v_x:
            break
        x += p_x // stride
    
    return coords

class BloodDataset(object):
    
    def __init__(self, train_lst):
        self.train_lst = train_lst
        with open(train_lst, 'r') as f:
            lines = f.readlines()
            self.subjects = [line.strip() for line in lines]     
    
    def sample(self, subject_num, patch_num=100, sample_3d=True):
        '''
        t0 = time.time()
        
        indices = range(len(self.subjects))
        random.shuffle(indices)
        indices = indices[:subject_num]
        
        subjects_data = []
        for idx in indices:
            subjects_data.append(load_data(BloodSampler if sample_3d else BloodSampler2D,
                                 self.subjects[idx], patch_num))
        
        results = []
        for data in subjects_data:
            dd = {}
            dd['image'] = data[0]
            dd['mask'] = data[1]
            dd['coords'] = data[2]
            results.append(dd)
        
        print 'sample need %f' % (time.time() - t0)
        return Patch3DLoader(results) if sample_3d else ImageLoader(results)
        '''
        para_dict = {"train_list": self.train_lst}
        if sample_3d:
            return BLOOD_SEG(para_dict, stage="train")
        else:
            ImageLoaderNPZ(self.train_lst)

class BloodSampler(object):
    
    def __init__(self, blood):
        
        self.blood = blood
        self.img_size = blood.shape
        self.patch_size = cfg.TRAIN.DATA.PATCH_SIZE
    
    def sample_random(self, k):
        
        coords = []
        '''
        a = np.where(self.blood == 255)
        num = len(a[0])
        PX, PY, PZ = self.patch_size
        sucess = 0
        for t in range(10000):
            i = random.randint(0, num-1)
            x0, x1 = a[0][i]-PX//2, a[0][i]+PX-PX//2
            y0, y1 = a[1][i]-PY//2, a[1][i]+PY-PY//2
            z0, z1 = a[2][i]-PZ//2, a[2][i]+PZ-PZ//2
            if x0 < 0 or x1 > self.img_size[0]:
                continue
            if y0 < 0 or y1 > self.img_size[1]:
                continue
            if z0 < 0 or z1 > self.img_size[2]:
                continue
            coords.append((x0, y0, z0))
            sucess += 1
            if sucess >= k:
                break
        '''
        x0, x1 = 0, self.img_size[0] - cfg.TRAIN.DATA.PATCH_SIZE[0]
        y0, y1 = 0, self.img_size[1] - cfg.TRAIN.DATA.PATCH_SIZE[1]
        z0, z1 = 0, self.img_size[2] - cfg.TRAIN.DATA.PATCH_SIZE[2]
        for i in range(k):
            x = random.randint(x0, x1)
            y = random.randint(y0, y1)
            z = random.randint(z0, z1)
            coords.append((x,y,z))
        return coords
            
    
    def sample_traverse(self, k):
        
        coords = get_patch_coords(self.patch_size, self.img_size, 2)
        random.shuffle(coords)
        return coords[:-1]

    def sample(self, k):
        
        coords = []
        coords.extend(self.sample_traverse(k))
        #coords.extend(self.sample_random(k))
        return coords
        
class BloodSampler2D(object):
    
    def __init__(self, blood):
        
        self.blood = blood
        self.img_size = blood.shape

    def sample(self, k):
        
        z = range(1, self.img_size[0]-1)
        #random.shuffle(z)
        coords = [(i, 0, 0) for i in z]
        return coords[:-1]


class ImageLoader(data.Dataset):
    
    def __init__(self, data_list):
    
        self.data_list = data_list
        self.num = 0
        self.list = []
        
        self.flip = False
        
        for idx, data in enumerate(data_list):
            self.list.extend([(idx, jdx) 
                              for jdx in range(len(data['coords']))])
    
    def __getitem__(self, index):
        
        data_i, coord_i = self.list[index]
        image = self.data_list[data_i]['image']
        mask = self.data_list[data_i]['mask']
       
        coord = self.data_list[data_i]['coords'][coord_i]
        
        image = image[coord[0]-1:coord[0]+2, :, :]
        
        mask = mask[coord[0]]
        
        #image = set_window_wl_ww(image)
        image = (image / 255.0) * 2.0 - 1.0
        
        if self.flip:
            flip_x = np.random.choice(2)*2 - 1
            flip_y = np.random.choice(2)*2 - 1
            image = image[::, ::flip_x, ::flip_y]
            mask = mask[::flip_x, ::flip_y]
        
        image = image.astype('float32')
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        return image, mask

    def __len__(self):
        return len(self.list)

class ImageLoaderNPZ(data.Dataset):
    def __init__(self, train_lst):
        with open(train_lst, 'r') as f:
            lines = f.readlines()
            self.npz_files = [line.strip() for line in lines]
        random.shuffle(self.npz_files)
        self.flip = False
    
    def __getitem__(self, index):
        npz_pth = self.npz_files[index]
        npz = np.load(npz_pth)
        img, gt = npz['img'], npz['gt']
        WL, WW = cfg.TRAIN.DATA.WL_WW
        img = set_window_wl_ww(img, WL, WW)
        img = (img / 255.0) * 2.0 - 1.0
        
        if self.flip:
            flip_x = np.random.choice(2)*2 - 1
            flip_y = np.random.choice(2)*2 - 1
            img = img[::, ::flip_x, ::flip_y]
            gt = gt[::flip_x, ::flip_y]
        
        img = img.astype('float32')
        img = torch.from_numpy(img)
        gt = torch.from_numpy(gt)
        
        return img, gt        
    
    def __len__(self):
        return len(self.npz_files)
                 
class PatchLoaderNPZ(data.Dataset):
    def __init__(self, train_lst):
        with open(train_lst, 'r') as f:
            lines = f.readlines()
            self.npz_files = [line.strip() for line in lines]
        random.shuffle(self.npz_files)
        
    def __getitem__(self, index):
        npz_pth = self.npz_files[index]
        npz = np.load(npz_pth)
        img, gt, hp = npz['img'], npz['gt'], npz['heatmap']

        #rotation
        #rotation_subject(img, gt)
        
        #flip
        #flip = np.random.choice(2)*2 - 1
        #img = img[:, ::flip, :].copy()
        #gt = gt[:, ::flip, :].copy()
        
        
        x1, y1, z1 = img.shape
        x0, y0, z0 = cfg.TRAIN.DATA.PATCH_SIZE
        
        x = random.randint(0, x1 - x0)
        y = random.randint(0, y1 - y0)
        z = random.randint(0, z1 - z0)
        
        img = img[x:x+x0, y:y+y0, z:z+z0]
        gt = gt[x:x+x0, y:y+y0, z:z+z0]
        hp = hp[x:x+x0, y:y+y0, z:z+z0]
        hp = hp.astype('float32')
        hp = hp[np.newaxis, :, :, :]
        WL, WW = cfg.TRAIN.DATA.WL_WW
        img = set_window_wl_ww(img, WL, WW)
        img = (img / 255.0) * 2.0 - 1.0
        
        img = img.astype('float32')
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)

        gt = torch.from_numpy(gt)
        hp = torch.from_numpy(hp)
        
        return {'img': img, 'gt': gt, 'heatmap': hp}
        #return img, gt        
    
    def __len__(self):
        return len(self.npz_files)
    
class ValidateImageLoader(data.Dataset):
    def __init__(self, val_list):
        
        with open(val_list, 'r') as f:
            lines = f.readlines()
            self.list = [line.strip() for line in lines]
        self.subjects = self.list[:10]
        self.images, self.masks = [], []
        for subject in self.subjects:
            image = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER,
                           '%s.nii.gz' % subject)).get_data()
            #blood = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER,
            #               '%s_blood.nii.gz' % subject)).get_data()
            blood = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER,
                            '%s.nii.gz' % subject)).get_data()
            
            image = np.transpose(image, (2, 0, 1))
            WL, WW = cfg.TRAIN.DATA.WL_WW
            image = set_window_wl_ww(image, WL, WW)
            
            blood = np.transpose(blood, (2, 0, 1))
            blood[blood > 128] = 1
            
            z = range(1, image.shape[0]-1)
            #random.shuffle(z)
            #z = z[:100]
            for i in z:
                img_tensor = np.zeros((3, image.shape[1], image.shape[2]), 'uint8')
                msk_tensor = np.zeros((image.shape[1], image.shape[2]), 'uint8')
                img_tensor[:, :, :] = image[i-1:i+2, :, :]
                msk_tensor[:,:] = blood[i]
                self.images.append(img_tensor)
                self.masks.append(msk_tensor)
    
    
    def __getitem__(self, index):
        
        image, mask = self.images[index], self.masks[index]
        image = (image / 255.0) * 2.0 - 1.0
        image = image.astype('float32')
        
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        return image, mask
    
    def __len__(self):
        return len(self.images) 
        
class ValidateImage3DLoader(data.Dataset):
    def __init__(self, val_list):
        
        self.heat_map = True
        
        
        with open(val_list, 'r') as f:
            lines = f.readlines()
            self.list = [line.strip() for line in lines]
            
        self.subjects = self.list[:20]
        self.images, self.masks, self.heatmaps = [], [], []
        
        for subject in self.subjects:
            image = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER,
                           '%s.nii.gz' % subject)).get_data()
            
            blood = nib.load(os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER,
                             '%s.nii.gz' % subject)).get_data()
            
            if self.heat_map:
                heatmap = nib.load(os.path.join(cfg.TRAIN.DATA.HEAT_MAP,
                                   '%s/heatmap.nii.gz' % subject)).get_data()
                heatmap = heatmap.astype('float32')
                heatmap = np.transpose(heatmap, (2, 0, 1))
            
            image = np.transpose(image, (2, 0, 1))
            WL, WW = cfg.TRAIN.DATA.WL_WW
            image = set_window_wl_ww(image, WL, WW)
            
            blood = np.transpose(blood, (2, 0, 1))
            blood[blood > 0] = 1
            
            #x0, x1 = 0, image.shape[0] - cfg.TRAIN.DATA.PATCH_SIZE[0]
            #y0, y1 = 0, image.shape[1] - cfg.TRAIN.DATA.PATCH_SIZE[1]
            #z0, z1 = 0, image.shape[2] - cfg.TRAIN.DATA.PATCH_SIZE[2]
            coords = get_patch_coords(cfg.TRAIN.DATA.PATCH_SIZE, image.shape, 1)
            for i in range(len(coords)):
                #x = random.randint(x0, x1)
                #y = random.randint(y0, y1)
                #z = random.randint(z0, z1)
                x, y, z = coords[i]
                img_tensor = np.zeros(cfg.TRAIN.DATA.PATCH_SIZE, 'uint8')
                msk_tensor = np.zeros(cfg.TRAIN.DATA.PATCH_SIZE, 'uint8')
                
                
                img_tensor[:, :, :] = image[x:x+cfg.TRAIN.DATA.PATCH_SIZE[0],
                                            y:y+cfg.TRAIN.DATA.PATCH_SIZE[1],
                                            z:z+cfg.TRAIN.DATA.PATCH_SIZE[2]]
                
                msk_tensor[:,:,:] = blood[x:x+cfg.TRAIN.DATA.PATCH_SIZE[0],
                                          y:y+cfg.TRAIN.DATA.PATCH_SIZE[1],
                                          z:z+cfg.TRAIN.DATA.PATCH_SIZE[2]]
                
                if self.heat_map:
                    hp_tensor = np.zeros(cfg.TRAIN.DATA.PATCH_SIZE, 'float32')
                    hp_tensor[:,:,:] = heatmap[x:x+cfg.TRAIN.DATA.PATCH_SIZE[0],
                                               y:y+cfg.TRAIN.DATA.PATCH_SIZE[1],
                                               z:z+cfg.TRAIN.DATA.PATCH_SIZE[2]]
                    
                    self.heatmaps.append(hp_tensor)
                
                self.images.append(img_tensor)
                self.masks.append(msk_tensor)
    
    
    def __getitem__(self, index):
        
        image, mask = self.images[index], self.masks[index]
        image = (image / 255.0) * 2.0 - 1.0
        image = image.astype('float32')
        
        image = image[np.newaxis, :, :, :]
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        if self.heat_map:
            heatmap = self.heatmaps[index]
            heatmap = heatmap[np.newaxis, :, :, :]
            heatmap = torch.from_numpy(heatmap)
            return {'img': image, 'gt': mask, 'heatmap': heatmap}
        
        return {'img': image, 'gt': mask}
        #return image, mask
    
    def __len__(self):
        return len(self.images) 
    
class ValidatePatch3DLoader(data.Dataset):
    def __init__(self, val_list):
        
        with open(val_list, 'r') as f:
            lines = f.readlines()
            self.list = [line.strip() for line in lines]
    
    def __getitem__(self, index):
        
        img_pth, gt_pth = self.list[index].split(' ')
        image = misc.imread(img_pth)
        image = image.reshape(image.shape[1], image.shape[1], image.shape[1])
        gt = misc.imread(gt_pth)
        gt = gt.reshape(gt.shape[1], gt.shape[1], gt.shape[1])
        gt[gt > 128] = 1
        
        aa = [(image.shape[i] - cfg.TRAIN.DATA.PATCH_SIZE[i])//2 for i in range(3)]
        image = image[aa[0]:aa[0]+cfg.TRAIN.DATA.PATCH_SIZE[0],
                      aa[1]:aa[1]+cfg.TRAIN.DATA.PATCH_SIZE[1],
                      aa[2]:aa[2]+cfg.TRAIN.DATA.PATCH_SIZE[2]]
        gt = gt[aa[0]:aa[0]+cfg.TRAIN.DATA.PATCH_SIZE[0],
                aa[1]:aa[1]+cfg.TRAIN.DATA.PATCH_SIZE[1],
                aa[2]:aa[2]+cfg.TRAIN.DATA.PATCH_SIZE[2]]
        
        image = (image / 255.0) * 2.0 - 1.0
        image = image.astype('float32')
        
        gt = torch.from_numpy(gt)
        
        image = image[np.newaxis, :, :, :]
        image = torch.from_numpy(image)
        
        return image, gt
    
    def __len__(self):
        return len(self.list)
    
class Patch3DLoader(data.Dataset):
    
    def __init__(self, data_list):
    
        self.data_list = data_list
        self.num = 0
        self.list = []
        self.patch_size = cfg.TRAIN.DATA.PATCH_SIZE
        
        self.flip = False
        
        for idx, data in enumerate(data_list):
            self.list.extend([(idx, jdx) 
                              for jdx in range(len(data['coords']))])
    
    def __getitem__(self, index):
        
        data_i, coord_i = self.list[index]
        image = self.data_list[data_i]['image']
        mask = self.data_list[data_i]['mask']
       
        coord = self.data_list[data_i]['coords'][coord_i]
        
        image = image[coord[0]:coord[0]+self.patch_size[0],
                      coord[1]:coord[1]+self.patch_size[1],
                      coord[2]:coord[2]+self.patch_size[2]]
        
        mask = mask[coord[0]:coord[0]+self.patch_size[0],
                    coord[1]:coord[1]+self.patch_size[1],
                    coord[2]:coord[2]+self.patch_size[2]]
        
        image = (image / 255.0) * 2.0 - 1.0
        
        if self.flip:
            flip_x = np.random.choice(2)*2 - 1
            flip_y = np.random.choice(2)*2 - 1
            flip_z = np.random.choice(2)*2 - 1
            image = image[::flip_x, ::flip_y, ::flip_z]
            mask = mask[::flip_x, ::flip_y, ::flip_z]
        
        image = image.astype('float32')
        image = image[np.newaxis, :, :, :]
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        return image, mask

    def __len__(self):
        return len(self.list)

class Volume2DLoader(data.Dataset):
    
    def __init__(self, subject):
        
        self.subject = subject
        
        #img = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER,
        #               '%s.nii.gz' % subject)).get_data()
        img = nib.load(subject).get_data()
        
        blood_pth = os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER, '%s.nii.gz' % subject)
        if os.path.exists(blood_pth):
            blood = nib.load(blood_pth).get_data()
            self.blood = np.transpose(blood, (2, 0, 1))
            self.blood[self.blood > 128] = 1
        else:
            self.blood = None

        self.img = np.transpose(img, (2, 0, 1))
        WL, WW = cfg.TRAIN.DATA.WL_WW
        self.img = set_window_wl_ww(self.img, WL, WW)
        
        self.coords = [(i, 0, 0) for i in range(1, self.img.shape[0]-1)]
    
    def __getitem__(self, index):
        x, y, z = self.coords[index]
        
        image = np.zeros((3, self.img.shape[1], self.img.shape[2]), 'uint8')
        image[:, :, :] = self.img[x-1:x+2, :, :]
        
        image = (image / 255.0) * 2.0 - 1.0
        
        image = image.astype('float32')
        image = torch.from_numpy(image)
        
        coord = np.array([[x, x + 1],
                          [y, y + self.img.shape[1]],
                          [z, z + self.img.shape[2]]])
        coord = torch.from_numpy(coord)        
        
        return image, coord
        
        
    def get_gt(self):
        return self.blood
    
    def volume_size(self):
        return self.img.shape
    
    def save(self, seg, res_pth, prob=False):
        
        img = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '%s.nii.gz' % self.subject))
        seg = np.transpose(seg, (1, 2, 0))
        seg = seg.astype('float32' if prob else 'uint8')
        affine = img.affine
        seg_img = nib.Nifti1Image(seg, affine)
        nib.save(seg_img, res_pth)
        
    def __len__(self):
        return len(self.coords)
    
class Volume3DLoader(data.Dataset):
    
    def __init__(self, subject):
        
        #cfg.TRAIN.DATA.NII_FOLDER = '/data2/pancw/data/cta_blood/image'
        #cfg.TRAIN.DATA.BLOOD_FOLDER = '/data2/pancw/data/cta_blood/blood'
        
        self.subject = subject
        img = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER,
                       '%s.nii.gz' % subject)).get_data()
        blood_pth = os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER, '%s.nii.gz' % subject)
        
        if os.path.exists(blood_pth):
            blood = nib.load(blood_pth).get_data()
            self.blood = np.transpose(blood, (2, 0, 1))
            self.blood[self.blood > 128] = 1
        else:
            self.blood = None

        self.img = np.transpose(img, (2, 0, 1))
        WL, WW = cfg.TRAIN.DATA.WL_WW
        self.img = set_window_wl_ww(self.img, WL, WW)
        
        self.patch_size = (24, 256, 256) #cfg.TRAIN.DATA.PATCH_SIZE
        self.coords = get_patch_coords(self.patch_size, self.img.shape, 2)
    
    def __getitem__(self, index):
        x, y, z = self.coords[index]
        
        img = self.img[x:x+self.patch_size[0],
                       y:y+self.patch_size[1],
                       z:z+self.patch_size[2]]
        
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)
        
        coord = np.array([[x, x + self.patch_size[0]],
                         [y, y + self.patch_size[1]],
                         [z, z + self.patch_size[2]]])
        coord = torch.from_numpy(coord)
                         
        return img, coord
    
    def get_gt(self):
        return self.blood
    
    def volume_size(self):
        return self.img.shape
    
    def save(self, seg, res_pth, prob=False):
        
        img = nib.load(os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '%s.nii.gz' % self.subject))
        seg = np.transpose(seg, (1, 2, 0))
        seg = seg.astype('float32' if prob else 'uint8')
        affine = img.affine
        seg_img = nib.Nifti1Image(seg, affine)
        nib.save(seg_img, res_pth)
        
    def __len__(self):
        return len(self.coords)
        

if __name__ == '__main__':
    
    generate_validation_dataset()