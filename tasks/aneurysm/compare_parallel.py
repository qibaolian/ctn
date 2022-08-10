import time
import os
import numpy as np
import nibabel as nib
from rotation_3d import getTransform, transform_point, transform_image
import SimpleITK as sitk

import multiprocessing as mp
import threading

from concurrent import futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

NII_FOLDER = '/data3/pancw/data/nii_file/'
ANEURYSM_FOLDER = '/data3/pancw/data/aneurysm_mask'

def func(subject):
    
    img = nib.load(os.path.join(NII_FOLDER, 
                   '%s.nii.gz' % subject)).get_data()
    aneurysm = nib.load(os.path.join(ANEURYSM_FOLDER,
                        '%s_aneurysm.nii.gz' % subject)).get_data()
    
    img = np.transpose(img, (2, 0, 1))
    aneurysm = np.transpose(aneurysm, (2, 0, 1))
    
    R_x, R_y, R_z = 90, 0, 0
    zoom = 1.0
    tt = getTransform(img.shape, R_x, R_y, R_z, (zoom, zoom, zoom))
    
    img[:] = transform_image(tt, img, 'linear')
    aneurysm[:] = transform_image(tt, aneurysm)
    
    return img, aneurysm

def func_list(s_list):
    for subject in s_list:
        func(subject)

result = []

def func2(subject):
    global result
    
    img = nib.load(os.path.join(NII_FOLDER, 
                   '%s.nii.gz' % subject)).get_data()
    aneurysm = nib.load(os.path.join(ANEURYSM_FOLDER,
                        '%s_aneurysm.nii.gz' % subject)).get_data()
    
    img = np.transpose(img, (2, 0, 1))
    aneurysm = np.transpose(aneurysm, (2, 0, 1))
    
    R_x, R_y, R_z = 90, 0, 0
    zoom = 1.0
    tt = getTransform(img.shape, R_x, R_y, R_z, (zoom, zoom, zoom))
    
    img[:] = transform_image(tt, img, 'linear')
    aneurysm[:] = transform_image(tt, aneurysm)
    
    
    result.append((img, aneurysm))

with open('/data3/pancw/data/patch/dataset/train/train_subjects.lst', 'r') as f:
        lines = f.readlines()
        subjects = [line.strip() for line in lines]

subjects = subjects[:8]
tt = time.time()
results = []
for subject in subjects:
    img, mask = func(subject)
    results.append((img, mask))
print ('normal need %.4f' % (time.time() - tt))

tt = time.time()
with ThreadPoolExecutor(8) as ex:
    res = ex.map(func, subjects)
print ('thread need %.4f' % (time.time() - tt))

tt = time.time()
with ProcessPoolExecutor() as ex:
    results = ex.map(func, subjects)
    #future_list = []
    #for subject in subjects:
    #    future = ex.submit(func, subject)
    #    future_list.append(future)
    #ex.shutdown()

#results = []
#for future in future_list:
#for future in futures.as_completed(future_list):
#    res = future.result()
#    results.append(res)
    
print ('process need %.4f' % (time.time() - tt))

for (img, mask) in results:
    print img.shape, mask.shape
'''
tt = time.time()
s_lists = [subjects[:8], subjects[8:16], subjects[16:24], subjects[24:32]]

for s_list in s_lists:
    func_list(s_list)
print ('normal need %.4f' % (time.time() - tt))

tt = time.time()
threads = [threading.Thread(target=func_list, args=(s_list,)) for s_list in s_lists]
for p in threads:
    p.start()
for p in threads:
    p.join()
print ('thread need %.4f' % (time.time() - tt))
'''

'''
tt = time.time()
q = mp.Queue()
processes = [mp.Process(target=func2, args=(subject,)) for subject in subjects]
for p in processes:
    p.start()
    
for p in processes:
    p.join()
    
print ('process need %.4f' % (time.time() - tt))
for (img, mask) in result:
    print img.shape, mask.shape
'''

'''
tt = time.time()
pool = mp.Pool(processes=4)
results = [pool.apply(func2, args=(subject,)) for subject in subjects]
print('pool need %.4f' % (time.time() - tt))
'''    
    
    
