import numpy as np
import pandas as pd
import os
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
import h5py
import re
import sys
from FPRpreproc_utils import saveNodule

#get trues from gt images
#get falses from pred images

gtDir = '/data/lung_seg/lung_nifti_files'
predDir = '/data/lung_seg/tempfrozen_dataforfpr/inference'
outDir = {'training': '/data/lung_seg/FPR/nodule_files/training', 'testing': '/data/lung_seg/FPR/nodule_files/testing', 'validation': '/data/lung_seg/FPR/nodule_files/validation'}
predIters = ['iter_11101', 'iter_11851', 'iter_12051', 'iter_13301', 'iter_11051'] #add more numbers
tpPredIters = ['iter_11101', 'iter_11051', 'iter_11851']
validationSet = list(np.arange(3,2000,10)) + list(np.arange(7000, 10000, 10)) + list(np.arange(15000,18000,10))
testingSet = []
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

#preprocess true nodules
s = np.ones((3,3,3))
gtnames = os.listdir(gtDir)

labels_names = [name for name in gtnames if seg_tag in name if int(re.findall(r'[0-9]+', name)[0]) in PIDList]
for label_name in labels_names:
    nibFile = nib.load(os.path.join(gtDir, label_name))
    labels = np.array(nibFile.dataobj).astype(np.int32)
    img_num = int(re.findall(r'[0-9]+', label_name)[0])
    vol_name = 'volume-{}.nii.gz'.format(img_num)
    nibFile = nib.load(os.path.join(gtDir, vol_name))
    volume = np.array(nibFile.dataobj).astype(np.float32)
    if np.sum(labels) == 0:
        continue
    else:
        labeledVol, nodCount = label(labels, s)
        for i in range(nodCount):
            noduleVol = (labeledVol == (i+1))
            saveNodule(noduleVol, volume, 1, testingSet, validationSet, outDir, total_nod_count)
            total_nod_count += 1
            '''
            x,y,z = np.nonzero(noduleVol)
            x1 = np.min(x)-5
            x2 = np.max(x)+5
            y1 = np.min(y)-5
            y2 = np.max(y)+5
            z1 = np.min(z)-5
            z2 = np.max(z)+5
            nodVol = (volume[x1:x2, y1:y2, z1:z2]).astype(np.float32)
            
            if total_nod_count in testingSet:
                split = 'testing'
            elif total_nod_count in validationSet:
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
            
            '''

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
                            saveNodule(prednoduleVol, img_vol, 0, testingSet, validationSet, outDir, total_nod_count)
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
                            saveNodule(prednoduleVol, img_vol, 0, testingSet, validationSet, outDir, total_nod_count)
                            total_nod_count += 1
                        elif (predIter in tpPredIters) and (2*maxIntersect/(np.sum(temp_gt_label_vol) + np.sum(prednoduleVol)) >= 0.3):
                            saveNodule(prednoduleVol, img_vol, 1, testingSet, validationSet, outDir, total_nod_count)
                            total_nod_count += 1
                            '''x,y,z = np.nonzero(prednoduleVol)
                            x1 = np.min(x)-5
                            x2 = np.max(x)+5
                            y1 = np.min(y)-5
                            y2 = np.max(y)+5
                            z1 = np.min(z)-5
                            z2 = np.max(z)+5
                            nodVol = (img_vol[x1:x2, y1:y2, z1:z2]).astype(np.float32)
                            
                            if total_nod_count in testingSet:
                                split = 'testing'
                            elif total_nod_count in validationSet:
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
                            '''
'''
x,y,z = np.nonzero(prednoduleVol)
x1 = np.min(x)-5
x2 = np.max(x)+5
y1 = np.min(y)-5
y2 = np.max(y)+5
z1 = np.min(z)-5
z2 = np.max(z)+5
nodVol = (img_vol[x1:x2, y1:y2, z1:z2]).astype(np.float32)

if total_nod_count in testingSet:
    split = 'testing'
elif total_nod_count in validationSet:
    split = 'validation'
else:
    split = 'training'

path_vol_save = os.path.join(outDir[split], '{}.h5'.format(total_nod_count))
with h5py.File(path_vol_save, 'w') as h5f:
    h5f.create_dataset('data', data=nodVol, dtype=np.float32)
path_label_save = os.path.join(outDir[split], '{}.txt'.format(total_nod_count))
with open(path_label_save, 'w+') as f:
    f.write('0')
    f.close()

total_nod_count += 1
'''
