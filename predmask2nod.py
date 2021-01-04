import numpy as np
import pandas as pd
import os
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
import h5py
import re
import sys
from predmask2nod_utils import saveNodule

#get trues from gt images
#get falses from pred images

gtDir = '/data/lung_seg/lung_nifti_files'
predDir = '/data/lung_seg/tempfrozen_dataforfpr/inference'
outDir = '/media/data_crypt_2/temp'
predIters = ['iter_22501'] #add more numbers
'''
gtDir = '/Volumes/Transcend/nifti_files/lung_nifti_files'
predDir = '/Users/ChaeyoungLee/Downloads/inference2'
outDir = {'training': '/Users/ChaeyoungLee/Documents/FPR/training', 'testing': '/Users/ChaeyoungLee/Documents/FPR/testing', 'validation': '/Users/ChaeyoungLee/Documents/FPR/validation'}
predIters = ['iter_13301']    #'iter_11551', 'iter_14701', 'iter_16001','iter_18301','iter_13751'] #add more numbers
testingSet = [] #list(np.arange(1,2000,10))
validationSet = [] #list(np.arange(3,2000,10))
'''
assert os.path.isdir(gtDir)
assert os.path.isdir(predDir)

for key in outDir:
    dirName = outDir[key]
    if not os.path.isdir(dirName):
        os.mkdir(dirName)

vol_tag = 'volume'
seg_tag = 'labels'

temp = os.listdir(os.path.join(predDir, predIters[0], 'prediction'))
#that predIter has to have all volumes in testing set!!!!
PIDList = [int(re.findall(r'[0-9]+', name)[0]) for name in temp]

total_nod_count = 0

s = np.ones((3,3,3))

#get false positives (class=0)
for predIter in os.listdir(predDir):
    if predIter in predIters:
        for predFile in os.listdir(os.path.join(predDir, predIter, 'prediction')):
            if '.nii.gz' in predFile:
                img_num = int(re.findall(r'[0-9]+', predFile)[0])
                print(img_num)
                pred_label_file = nib.load(os.path.join(predDir, predIter, 'prediction', predFile))
                pred_label_vol = np.array(pred_label_file.dataobj).astype(np.int32)
                img_vol_name = 'volume-{}.nii.gz'.format(img_num)
                img_file = nib.load(os.path.join(gtDir, img_vol_name))
                img_vol = np.array(img_file.dataobj).astype(np.float32)
                gt_label_file = nib.load(os.path.join(gtDir, 'labels-{}.nii.gz'.format(img_num)))
                gt_label_vol = np.array(gt_label_file.dataobj).astype(np.int32)
                if np.sum(pred_label_vol) == 0:
                    continue
                else:
                    predLabeledVol, prednodCount = label(pred_label_vol, s)
                    gtLabeledVol, gtnodCount = label(gt_label_vol, s)
                    for i in range(prednodCount):
                        prednoduleVol = (predLabeledVol == (i+1))
                        if gtnodCount == 0:
                            print(str('gtnodcount is 0'))
                            saveNodule(prednoduleVol, img_vol, 0, outDir, total_nod_count)
                            total_nod_count += 1
                            continue
                        intersects = []
                        for a in range(gtnodCount):
                            gtnoduleVol = (gtLabeledVol == (a+1))
                            numIntersect = np.sum(gtnoduleVol & prednoduleVol)
                            intersects.append([numIntersect, (a+1)])
                        temp = [intersects[a][0] for a in range(len(intersects))]
                        maxIntersect = max(temp)
                        index = intersects[temp.index(maxIntersect)][1]
                        temp_gt_label_vol = (gtLabeledVol == index)
                        if 2*maxIntersect/(np.sum(temp_gt_label_vol) + np.sum(prednoduleVol)) < 0.15:
                            saveNodule(prednoduleVol, img_vol, 0, outDir, total_nod_count)
                            total_nod_count += 1
                        elif (2*maxIntersect/(np.sum(temp_gt_label_vol) + np.sum(prednoduleVol)) >= 0.15):
                            saveNodule(prednoduleVol, img_vol, 1, outDir, total_nod_count)
                            total_nod_count += 1
