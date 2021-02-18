import nibabel as nib
import numpy as np
import os
import scipy.ndimage
import re

#on jupiter -- did preprocessing for training & validation
#on saturn -- did preprocessing for testing

'''
#on jupiter
predDir = '/data/lung_inference/inference2'
outDir = '/data/lung_resized_inference'
gtDir = '/data/improved_res_lung/improved_res_nifti_files'
'''

#on jupiter for testing
predDir = '/data/lung_inference/inference_for_test'
outDir = '/data/lung_resized_inference/testing_iters'
gtDir = '/data/improved_res_lung/improved_res_nifti_files'

for predIter in os.listdir(predDir):
    if 'iter_' in predIter:
        print(predIter)
        alr = os.listdir(os.path.join(outDir, predIter, 'prediction'))
        for predFile in os.listdir(os.path.join(predDir, predIter, 'prediction')):
            if ('.nii.gz' in predFile) and (predFile not in alr):
                try:
                    img_num = int(re.findall(r'[0-9]+', predFile)[0])
                    print(img_num)
                    label_file = nib.load(os.path.join(predDir, predIter, 'prediction', predFile))
                    vol = np.array(label_file.dataobj).astype(np.int32)
                    gt = nib.load(os.path.join(gtDir, 'labels-{}.nii.gz'.format(img_num)))
                    new_shape = np.array(gt.dataobj.shape[:3])
                    old_shape = np.array(vol.shape[:3])
                    resize_factor = new_shape/old_shape
                    resized_image = scipy.ndimage.zoom(vol, resize_factor)
                    img = nib.Nifti1Image(resized_image, None)
                    nib.save(img, os.path.join(outDir,predIter,'prediction',predFile))
                except:
                    print('error!')
                    continue
