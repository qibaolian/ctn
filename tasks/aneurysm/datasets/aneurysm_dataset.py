from utils.config import cfg
from tasks.aneurysm.datasets.base_dataset import BaseDataset
from tasks.aneurysm.datasets.data_utils import *

import numpy as np
import torch
import nibabel as nib
import os


class ANEURYSM_SEG(BaseDataset):
    def __init__(self, para_dict, stage):
        super(ANEURYSM_SEG, self).__init__(para_dict, stage)

    def train_init(self):
        self.image = self.para_dict.get("image", None)
        self.gt = self.para_dict.get("gt", None)
        self.flip = False
        self.num = len(self.image)

    def test_init(self):
        self.subject = self.para_dict.get("subject", None)
        self.img = nib.load(os.path.join(cfg.TEST.DATA.NII_FOLDER,
                                         "{}.nii.gz".format(self.subject))).get_data()
        self.img = np.transpose(self.img, (2, 0, 1))
        self.patch_size = cfg.TEST.DATA.PATCH_SIZE
        self.coords = get_patch_coords(self.patch_size, self.img.shape)
        self.num = len(self.coords)

    def train_load(self, index):
        image, gt = self.image[index], self.gt[index]
        image = (image / 255.0) * 2.0 - 1.0
        label = 1 if gt.sum() > 0 else 0

        if self.flip:
            flip_x = np.random.choice(2) * 2 - 1
            flip_y = np.random.choice(2) * 2 - 1
            flip_z = np.random.choice(2) * 2 - 1
            image = image[::flip_x, ::flip_y, ::flip_z]
            gt = gt[::flip_x, ::flip_y, ::flip_z]

        image = image.astype('float32')
        image = image[np.newaxis, :, :, :]
        image = torch.from_numpy(image)
        gt = torch.from_numpy(gt)
        data = {'img': image, 'gt': gt, 'label': label}
        return data

    def test_load(self, index):
        x, y, z = self.coords[index]

        img = self.img[x:x + self.patch_size[0],
              y:y + self.patch_size[1],
              z:z + self.patch_size[2]]
        WL, WW = cfg.TRAIN.DATA.WL_WW
        img = set_window_wl_ww(img, WL, WW)
        img = (img / 255.0) * 2.0 - 1.0
        img = img.astype('float32')
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)
        coord = np.array([[x, x + self.patch_size[0]],
                          [y, y + self.patch_size[1]],
                          [z, z + self.patch_size[2]]])
        coord = torch.from_numpy(coord)
        return img, coord

    def volume_size(self):
        return self.img.shape

    def save(self, seg, folder):
        path = os.path.join(folder, '%s_seg.nii.gz' % self.subject)

        img = nib.load(os.path.join(cfg.TEST.DATA.NII_FOLDER, '%s.nii.gz' % self.subject))
        seg = np.transpose(seg, (1, 2, 0))
        # seg = seg.astype('uint8')
        affine = img.affine
        seg_img = nib.Nifti1Image(seg, affine)
        nib.save(seg_img, path)
