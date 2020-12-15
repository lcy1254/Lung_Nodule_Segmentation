import numpy as np
import pandas as pd
import os
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
import h5py
import re
import sys

#get trues from gt images
#get falses from pred images

gtDir = 'data/lung_seg/lung_nifti_files'
predDir = 'data/lung_seg/inference'
outDir = {'training': 'data/lung_seg/FPR/nodule_files/training', 'testing': 'data/lung_seg/FPR/nodule_files/testing', 'validation': 'data/lung_seg/FPR/nodule_files/validation'}
testingSet = []
validationSet = []

assert os.path.isdir(gtDir)
assert os.path.isdir(predDir)

for key in outDir:
    dirName = outDir[key]
    if not os.path.isdir(dirName):
        os.mkdir(dirName)

vol_tag = 'volume'
seg_tag = 'labels'

total_nod_count = 0

#preprocess true nodules
s = np.ones((3,3,3))
gtnames = os.listdir(gtDir)
labels_names = [name for name in gtnames if seg_tag in name]
for label_name in labels_names:
    label_file = nib.load(os.path.join(gtDir, label_name))
    label_vol = np.array(label_file.dataobj).astype(np.int32)
    img_num = (re.findall(r'[0-9]+', label_name)[0])
    vol_name = 'volume-{}.nii.gz'.format(img_num)
    img_file = nib.load(os.path.join(gtDir, vol_name))
    img_vol = np.array(img_file.dataobj).astype(np.float32)
    if np.sum(label_vol) == 0:
        continue
    else:
        labeledVol, nodCount = label(label_vol, s)
        for i in range(nodCount):
            tempVol = (labeledVol == (i+1))
                x,y,z = np.nonzero(tempVol)
                x1 = np.min(x)-5
                x2 = np.max(x)+5
                y1 = np.min(y)-5
                y2 = np.max(y)+5
                z1 = np.min(z)-5
                z2 = np.max(z)+5
                nodVol = (img_vol[x1:x2, y1:y2, z1:z2]).astype(np.float32)
                
                if img_num in testingSet:
                    split = 'testing'
                elif img_num in validationSet:
                    split = 'validation'
                else:
                    split = 'training'
                
                path_vol_save = os.path.join(outDir[split], '{}.h5'.format(total_nod_count))
                with h5py.File(path_vol_save, 'w') as h5f:
                    h5f.create_dataset('data', data=nodVol, dtype=np.float32)
                path_label_save = os.path.join(outDir[split], '{}.txt'.format(total_nod_count))
                with open(path_label_save, 'w+') as f:
                    f.write('1')
                    f.close()
                
                total_nod_count += 1


