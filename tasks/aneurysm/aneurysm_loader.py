# -*- coding=utf-8 -*-
import os
import torch
from torch.utils import data
import math
import random
import nibabel as nib
from functools import reduce
import time
from skimage import measure
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tasks.aneurysm.rotation_3d import getTransform, transform_point, transform_image
from tasks.aneurysm.datasets.data_utils import *
from tasks.aneurysm.datasets.aneurysm_dataset import ANEURYSM_SEG
from utils.config import cfg
from utils.tools.util import progress_monitor



AUG_ROTATION = True


def read_data(subject):
    """
    dataset organization:
    eg: subject-0649385
    - 0649385.nii.gz
    - 0649385_blood.nii.gz
    - 0649385_aneurysm.nii.gz / 0649385_mask.nii.gz
    """
    img_path = os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '%s.nii.gz' % subject)
    img = nib.load(img_path).get_data()

    aneurysm_path = os.path.join(cfg.TRAIN.DATA.ANEURYSM_FOLDER, '%s_aneurysm.nii.gz' % subject)
    aneurysm_path2 = os.path.join(cfg.TRAIN.DATA.ANEURYSM_FOLDER, '%s_mask.nii.gz' % subject)
    if os.path.exists(aneurysm_path):
        #print(aneurysm_path)
        aneurysm = nib.load(aneurysm_path).get_data()
    elif os.path.exists(aneurysm_path2):
        #print(aneurysm_path2)
        aneurysm = nib.load(aneurysm_path2).get_data()
    else:
        #print('=======0000=======')
        aneurysm = np.zeros(img.shape, 'uint8')

    blood_path = os.path.join(cfg.TRAIN.DATA.BLOOD_FOLDER, '%s_blood.nii.gz' % subject)

    if os.path.exists(blood_path):
        blood = nib.load(blood_path).get_data()
    else:
        blood = np.zeros(img.shape, 'uint8')

    # transpose dimensions of an array:
    # xyz=> zxy
    img = np.transpose(img, (2, 0, 1))
    aneurysm = np.transpose(aneurysm, (2, 0, 1))
    blood = np.transpose(blood, (2, 0, 1))

    return img, aneurysm, blood


def rotation_subject(img, aneurysm, blood, bbox_list, rotaton, zoom=1.0):
    R_x, R_y, R_z = rotaton

    tt = getTransform(img.shape, R_x, R_y, R_z, (zoom, zoom, zoom))
    tt_inv = tt.GetInverse()

    '''
    t0 = time.time()
    new_bbox = []
    for bbox in bbox_list:
        x, y, z, w, h, d = bbox
        mm = aneurysm[x:x+w, y:y+h, z:z+d]
        aa = np.where(mm > 0)
        x_min, y_min, z_min =  1000,  1000,  1000
        x_max, y_max, z_max = -1000, -1000, -1000
        for i in range(len(aa[0])):
            xx, yy, zz = transform_point(tt_inv, np.array([x+aa[0][i], y+aa[1][i], z+aa[2][i]], dtype='float64'))
            x_min = min(x_min, xx)
            y_min = min(y_min, yy)
            z_min = min(z_min, zz)
            x_max = max(x_max, xx)
            y_max = max(y_max, yy)
            z_max = max(z_max, zz)
        x_min, y_min, z_min = int(x_min), int(y_min), int(z_min)
        x_max, y_max, z_max = int(math.ceil(x_max)), int(math.ceil(y_max)), int(math.ceil(z_max))
        
        assert(x_min >=0 and y_min >=0 and z_min >=0 and \
               x_max <  img.shape[0] and y_max < img.shape[1] and z_max < img.shape[2])
        
        new_bbox.append([x_min, y_min, z_min, x_max+1-x_min, y_max+1-y_min, z_max+1-z_min])
   
    print(new_bbox, '%.4f' % (time.time() - t0))
    '''

    new_bbox = []
    for bbox in bbox_list:
        x, y, z, w, h, d = bbox
        if w == 1 or h == 1 or d == 1:  # 单层数据
            continue
        mm = aneurysm[x:x + w, y:y + h, z:z + d]
        labels = measure.label(mm)
        props = measure.regionprops(labels)
        if len(props) == 0:
            print('bbox error', bbox)
            continue

        prop = reduce(lambda x, y: x if x['area'] > y['area'] else y, props)
        coords = np.array(list(
            map(lambda xx: transform_point(tt_inv, np.array([x + xx[0], y + xx[1], z + xx[2]], dtype='float64')),
                prop.coords)))
        x_min, x_max = int(np.min(coords[:, 0])), int(math.ceil(np.max(coords[:, 0])))
        y_min, y_max = int(np.min(coords[:, 1])), int(math.ceil(np.max(coords[:, 1])))
        z_min, z_max = int(np.min(coords[:, 2])), int(math.ceil(np.max(coords[:, 2])))

        # assert(x_min >=0 and y_min >=0 and z_min >=0 and \
        #       x_max <  img.shape[0] and y_max < img.shape[1] and z_max < img.shape[2])
        x_min, x_max = max(0, x_min), min(x_max, img.shape[0] - 1)
        y_min, y_max = max(0, y_min), min(y_max, img.shape[1] - 1)
        z_min, z_max = max(0, z_min), min(z_max, img.shape[2] - 1)

        new_bbox.append([x_min, y_min, z_min, x_max + 1 - x_min, y_max + 1 - y_min, z_max + 1 - z_min])

    img[:] = transform_image(tt, img, 'linear')
    aneurysm[:] = transform_image(tt, aneurysm)
    blood[:] = transform_image(tt, blood)

    return new_bbox


def sample_subject(subject, image, aneurysm, blood, bbox_list, patch_num=100, rotation=True):
    """ positive and negative PATCH """
    if rotation:
        R_x = (np.random.choice(2) * 2 - 1) * np.random.choice(30)
        R_y = (np.random.choice(2) * 2 - 1) * np.random.choice(10)
        R_z = (np.random.choice(2) * 2 - 1) * np.random.choice(10)
        # zoom = 1.0
        zoom = np.random.uniform(0.95, 1.05)
        bbox_list = rotation_subject(image, aneurysm, blood, bbox_list, (R_x, R_y, R_z), zoom)

    sampler = AneurysmSampler(subject, bbox_list, image, aneurysm, blood)
    return sampler.sample(patch_num)


def sampe_subjects(subjects, aneurysm_bbox, patch_num):
    img_lst, gt_lst = [], []
    for subject in subjects:
        image, aneurysm, blood = read_data(subject)  # origin data
        img, gt = sample_subject(subject, image, aneurysm, blood, aneurysm_bbox[subject], patch_num)
        img_lst.append(img)
        gt_lst.append(gt)

    return np.concatenate(img_lst), np.concatenate(gt_lst)


def get_aneurysm_bbox():
    aneurysm_bbox = {}
    with open(cfg.TRAIN.DATA.ANEURYSM_BBOX, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vv = line.strip().split(' ')
            if vv[0] not in aneurysm_bbox:
                aneurysm_bbox[vv[0]] = []
            aneurysm_bbox[vv[0]].append([int(v) for v in vv[1:]])
    return aneurysm_bbox


class AneurysmDataset(object):

    def __init__(self, train_lst):

        self.train_lst = train_lst
        with open(self.train_lst, 'r') as f:
            lines = f.readlines()
            self.subjects = [line.strip() for line in lines]

        self.aneurysm_bbox = get_aneurysm_bbox()
        for subject in self.subjects:
            if subject not in self.aneurysm_bbox:
                self.aneurysm_bbox[subject] = []

        self.ex = None
        self.objs = []

    def asyn_sample(self, subject_num=100, patch_num=100, max_workers=20):

        indices = list(range(len(self.subjects)))
        random.shuffle(indices)
        indices = indices[:subject_num]

        monitor = progress_monitor(total=len(indices))
        subjects = [self.subjects[idx] for idx in indices]
        self.ex = ProcessPoolExecutor(max_workers=max_workers)
        self.objs = []
        # step = int(math.ceil((subject_num / 20)))
        step = 1
        for i in range(0, len(subjects), step):
            future = self.ex.submit(sampe_subjects, subjects[i:i + step], self.aneurysm_bbox, patch_num)
            future.add_done_callback(fn=monitor)
            self.objs.append(future)
        print('data processing in async ...')

    def get_data_loader(self):

        self.ex.shutdown(wait=True)
        img_lst, gt_lst = [], []
        for obj in self.objs:
            # img, gt = obj.result()
            img, gt = obj.result()
            img_lst.append(img)
            gt_lst.append(gt)

        t0 = time.time()
        img, gt = np.concatenate(img_lst), np.concatenate(gt_lst)
        print('concate need %.4f' % (time.time() - t0))
        para_dict = {"image": img, "gt": gt}
        return ANEURYSM_SEG(para_dict, "train")


class AneurysmSampler(object):

    def __init__(self, subject, bbox_list, image, aneurysm, blood, is_neck=True):
        '''传入一组数据对应的图像, 动脉瘤&血管MASK'''
        self.subject = subject
        self.image = image
        self.aneurysm = aneurysm
        self.blood = blood
        self.img_size = blood.shape
        self.patch_size = cfg.TRAIN.DATA.PATCH_SIZE
        self.offset = cfg.TRAIN.DATA.PATCH_OFFSET
        self.bbox_list = bbox_list
        self.is_neck = is_neck

    def sample_positive(self, bbox, k):
        '''一组位置处'''
        x_0, y_0, z_0, w, h, d = bbox
        x_1, y_1, z_1 = x_0 + w, y_0 + h, z_0 + d

        # 调整PATCH大小的原因: 默认PATCH小于病灶的情况
        p_size = (
            max(self.patch_size[0], w + 2 * self.offset[0]),
            max(self.patch_size[1], h + 2 * self.offset[1]),
            max(self.patch_size[2], d + 2 * self.offset[2])
        )
        # p_size = self.patch_size

        # 前提: PATCH大小必须大于病灶的尺寸
        # box的**右下**位置为起点,计算所切PATCH的**左上**边界阈值>=0
        x0 = max(0, x_1 + self.offset[0] - p_size[0])
        y0 = max(0, y_1 + self.offset[1] - p_size[1])
        z0 = max(0, z_1 + self.offset[2] - p_size[2])
        # box的**左上**坐标位置为起点,计算(更新)切PATCH的**左上**边界阈值
        x1 = max(0, min(x_0 - self.offset[0], self.img_size[0] - p_size[0]))
        y1 = max(0, min(y_0 - self.offset[1], self.img_size[1] - p_size[1]))
        z1 = max(0, min(z_0 - self.offset[2], self.img_size[2] - p_size[2]))

        if x0 > x1 or y0 > y1 or z0 > z1:
            print(x_0, x_1, y_0, y_1, z_0, z_1)
            print(self.subject, 'error')
            return [], []

        img_lst, gt_lst = [], []
        for i in range(k):
            # 基于上一步计算的起始点阈值区间, 随机选择一处进行采样
            x = random.randint(x0, x1)
            y = random.randint(y0, y1)
            z = random.randint(z0, z1)

            img = self.image[x:x + p_size[0], y:y + p_size[1], z:z + p_size[2]]
            aneu = self.aneurysm[x:x + p_size[0], y:y + p_size[1], z:z + p_size[2]]
            gt = np.zeros(img.shape, 'uint8')

            b1 = x, y, z, p_size[0], p_size[1], p_size[2]
            for b2 in self.bbox_list:
                if bbox3d_contain(b1, b2):
                    xx = [b2[i] - b1[i] for i in range(3)]
                    yy = [xx[i] + b2[i + 3] for i in range(3)]
                    gt[xx[0]:yy[0], xx[1]:yy[1], xx[2]:yy[2]] = \
                        aneu[xx[0]:yy[0], xx[1]:yy[1], xx[2]:yy[2]]
            WL, WW = cfg.TRAIN.DATA.WL_WW
            img = set_window_wl_ww(img, WL, WW)
            gt[gt > 0] = 1

            # 调整了PATCH默认尺寸的情况, 随机位置切(一)片
            x = random.randint(0, p_size[0] - self.patch_size[0])
            y = random.randint(0, p_size[1] - self.patch_size[1])
            z = random.randint(0, p_size[2] - self.patch_size[2])

            img = img[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]]
            gt = gt[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]]

            img_lst.append(img[np.newaxis, :, :, :])
            gt_lst.append(gt[np.newaxis, :, :, :])

        return img_lst, gt_lst

    def sample_negative_random(self, k):

        max_try_times = 10000
        neg_coords = []
        neg_num = 0
        for try_iter in range(max_try_times):

            x = random.randint(0, self.img_size[0] - self.patch_size[0])
            y = random.randint(0, self.img_size[1] - self.patch_size[1])
            z = random.randint(0, self.img_size[2] - self.patch_size[2])

            b1 = [x, y, z, self.patch_size[0],
                  self.patch_size[1], self.patch_size[2]]
            r = 0
            for b2 in self.bbox_list:
                r = max(bbox3d_ratio(b1, b2), r)

            if r < 0.5:
                if self.is_neck:  # 头颈数据数据, 限制特定层面范围采样
                    coord = filter_negative_coords([(x, y, z)], layer_total=self.img_size[0])
                    if coord:
                        neg_coords.append(coord[0])
                        neg_num += 1
                else:
                    neg_coords.append((x, y, z))
                    neg_num += 1

            if neg_num >= k:
                break

        return neg_coords

    def sample_negative_blood_random(self, k):

        coords = []
        a = np.where(self.blood > 0)
        num = len(a[0])
        if num == 0:
            return coords

        PX, PY, PZ = self.patch_size
        sucess = 0
        for t in range(20000):
            i = random.randint(0, num - 1)

            x0 = a[0][i] - PX // 2 + (np.random.choice(2) * 2 - 1) * np.random.choice(PX // 4)
            y0 = a[1][i] - PY // 2 + (np.random.choice(2) * 2 - 1) * np.random.choice(PY // 4)
            z0 = a[2][i] - PZ // 2 + (np.random.choice(2) * 2 - 1) * np.random.choice(PZ // 4)

            x0 = min(max(0, x0), self.img_size[0] - PX)
            y0 = min(max(0, y0), self.img_size[1] - PY)
            z0 = min(max(0, z0), self.img_size[2] - PZ)

            b1 = [x0, y0, z0, self.patch_size[0],
                  self.patch_size[1], self.patch_size[2]]
            r = 0
            for b2 in self.bbox_list:
                r = max(bbox3d_ratio(b1, b2), r)

            if r < 0.2:
                if self.is_neck:  # 头颈数据数据, 限制特定层面范围采样
                    coord = filter_negative_coords([(x0, y0, z0)], layer_total=self.img_size[0])
                    if coord:
                        coords.append(coord[0])
                        sucess += 1
                else:
                    coords.append((x0, y0, z0))
                    sucess += 1

            if sucess >= k:
                break

        return coords

    def sample_negative_traverse(self, k):
        """遍历方式负样本采样"""
        coords = get_patch_coords(self.patch_size, self.img_size, 2)
        if self.is_neck:  # 头颈数据数据, 限制特定层面范围采样max(300, 1/2 * layer * total)
            coords = filter_negative_coords(coords, layer_total=self.img_size[0])
        random.shuffle(coords)
        neg_coords = []
        neg_num = 0
        for idx, coord in enumerate(coords):
            b1 = [coord[0], coord[1], coord[2],
                  self.patch_size[0], self.patch_size[1], self.patch_size[2]]
            r = 0
            for b2 in self.bbox_list:
                r = max(bbox3d_ratio(b1, b2), r)

            if r < 0.2:
                neg_coords.append(coord)
                neg_num += 1

            if neg_num >= k:
                break

        return neg_coords

    def sample(self, k):
        '''正负样本的PATCH 采样 , 采样数k'''
        img_lst, gt_lst = [], []
        for bbox in self.bbox_list:
            img, gt = self.sample_positive(bbox, k)
            img_lst.extend(img)
            gt_lst.extend(gt)

        neg_num = max(k, len(img_lst))

        neg_coords = []
        neg_coords.extend(self.sample_negative_traverse(neg_num // 2))
        # print('0#########################', len(neg_coords))
        blood_coords = self.sample_negative_blood_random(neg_num - neg_num // 2)
        if len(blood_coords) > 0:
            # print("====sample_negative_blood_random====", len(neg_coords))
            neg_coords.extend(blood_coords)
        else:
            # print("****sample_negative_random******", len(neg_coords))
            neg_coords.extend(self.sample_negative_random(neg_num - neg_num // 2))

        for coord in neg_coords:
            x, y, z = coord
            img = self.image[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]]
            WL, WW = cfg.TRAIN.DATA.WL_WW
            img = set_window_wl_ww(img, WL, WW)
            gt = np.zeros(img.shape, 'uint8')

            img_lst.append(img[np.newaxis, :, :, :])
            gt_lst.append(gt[np.newaxis, :, :, :])

        return np.concatenate(img_lst), np.concatenate(gt_lst)


class ValidatePatch3DLoaderMemory(object):

    def __init__(self, val_list):

        with open(val_list, 'r') as f:
            lines = f.readlines()
            patch_list = [line.strip() for line in lines]
            # patch_list = [line.strip() for line in lines][:2] # for fast train

        def load_npz_data(pth_list):
            img_list, gt_list, label_list = [], [], []
            for npz_pth in pth_list:
                label = 1 if 'positive' in npz_pth else 0
                npz = np.load(npz_pth)
                img, gt = npz['img'], npz['gt']
                WL, WW = cfg.TRAIN.DATA.WL_WW
                img = set_window_wl_ww(img, WL, WW)
                img = (img / 255.0) * 2.0 - 1.0

                img_list.append(img[np.newaxis, np.newaxis, :, :, :].astype('float32'))
                gt_list.append(gt[np.newaxis, :, :, :])
                label_list.append(label)

            return img_list, gt_list, label_list

        self.ex = ThreadPoolExecutor()
        self.objs = []
        for i in range(0, len(patch_list), 200):
            self.objs.append(self.ex.submit(load_npz_data, patch_list[i:i + 200]))

        self.inited = False

    def get_tensor(self):

        if self.inited == False:
            self.ex.shutdown(wait=True)
            img_list, gt_list, label_list = [], [], []
            for obj in self.objs:
                data = obj.result()
                img, gt, label = data
                img_list.extend(img)
                gt_list.extend(gt)
                label_list.extend(label)

            img = np.concatenate(img_list)
            gt = np.concatenate(gt_list)
            label = np.array(label)
            self.img = torch.from_numpy(img)
            self.gt = torch.from_numpy(gt)
            self.label = torch.from_numpy(label)
            self.inited = True

        return self.img, self.gt, self.label


class ValidatePatch3DLoader(data.Dataset):
    def __init__(self, val_list):
        with open(val_list, 'r') as f:
            lines = f.readlines()
            self.list = [line.strip() for line in lines][:1000]

    def __getitem__(self, index):
        npz_pth = self.list[index]
        label = 1 if 'positive' in npz_pth else 0
        npz = np.load(npz_pth)

        img, gt = npz['img'], npz['gt']
        WL, WW = cfg.TRAIN.DATA.WL_WW
        img = set_window_wl_ww(img, WL, WW)
        img = (img / 255.0) * 2.0 - 1.0

        img = img.astype('float32')
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)

        gt = torch.from_numpy(gt)

        return {'img': img, 'gt': gt, 'label': label}

    def __len__(self):
        return len(self.list)

