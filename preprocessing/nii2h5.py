"""
!!!!! code from https://github.com/bbrister !!!!!
https://github.com/bbrister/CTH_seg_private/blob/master/nii2h5.py

        nii2h5.py

        DESCRIPTION
                This script converts a Nifti (.nii.gz) dataset into h5 format,
                separating the training and validation sets. This is used to
                prepare a dataset for the CTH_seg deep neural network semantic
                segmentation repository.


                To use this script for an existisng dataset, simply call it
                with one of the arguments below.

                To add a new dataset:
                        1. Make sure your data consists of pairs of Nifti files
                                volume-XXX.nii.gz
                                labels-XXX.nii.gz
                        where XXX is a number identifying that data pair. The
                        first file stores the image information, possibly in multiple 
                        channels as a 4D image volume, while the second stores the
                        3D class labels. Valid class labels are in the range 0-NUM_CLASS-1,
                        while negative labels are ignored during training.
                        2. Configure path_nii to be to the directory of the Nifti data
                        3. Configure path_h5 to be the directory of the output training
                        and validation data (see examples)
                        4. Configure validationSet to be a list of the data 
                        points which go in the validation data. These are the
                        numbers XXX above.
        USAGE
                Call this script with the command.
                        python nii2h5.py [mode]
                where mode is one of the following datasets:
                        organ
                        pet-ct-lung
                        liver_lesion
                        baselines-1mm
                        ct_lung
         
"""
import h5py
import numpy as np
import os
from os import listdir, mkdir, remove
from os.path import join, isdir, isfile
import nibabel as nib
import re
import itertools
import sys 

import matplotlib.pyplot as plt

"""
        -------------------CONFIGURATION---------------------
"""

mode = sys.argv[1] 

tag = '' # Add this to the file names
petCtMode = False # If true, loads multiple files
lungMode = False # If true, use multiple volumes per study

if mode == 'organ': # CT organs
        path_nii = '/data/organ/labels-3mm'
        path_h5 = {'training' : '/data/organ/training',
                   'validation': '/data/organ/validation'}
        validationSet = [x for x in itertools.chain(range(17), range(20, 23))] + [134]
elif mode == 'pet-ct-lung':
        petCtMode = True
        lungMode = True
        path_nii = '/data/pet-ct-vol/lung'
        path_h5 = {'training': os.path.join(path_nii, 'training'),
                   'validation': os.path.join(path_nii, 'validation')}
        validationSet = list(range(25))
elif mode == 'liver_lesion': # Liver lesions
        path_nii = '/data/liver_lesion/lesions-1mm'
        path_h5 = {'training' : '/data/liver_lesion/training',
                   'validation': '/data/liver_lesion/validation'}
        validationSet = list(range(20)) # Pick the first 20 volumes as validation
        # Note: this was previously 10, changed to improve stability
elif mode == 'baselines-1mm':
        path_nii = '/data/baselines-1mm'
        path_h5 = {'training' : '/data/liver_lesion/training',
                   'validation': '/data/liver_lesion/validation'}
        validationSet = [] # Don't use these for validation
        tag = 'mint-baseline'
elif mode == 'ct_lung': 
        path_nii = '/data/lung_seg/lung_nifti_files'
        path_h5 = {'training' : '/data/lung_seg/training_2',
                   'validation': '/data/lung_seg/validation_2'}
        validationSet = list(np.arange(1, 900, 3))
        validationSet.extend(list(range(901, 1000)))
        print(validationSet)
        dfsdfsd
else:
        raise ValueError('Unrecognized mode: ' + mode)

print('Validation indices:')
print(validationSet)
# Ensure the output directories exist
for key in path_h5:
        dirName = path_h5[key]
        if not os.path.isdir(dirName):
            os.makedirs(dirName)

"""
        -------------------PROCESSING---------------------
"""

# Substrings to differentiate between volumes and segmentations
vol_tag = 'pet' if petCtMode else 'volume'
seg_tag = 'labels'
voronoi_tag = 'voronoi'
fuzzy_weights_tag = 'fuzzyWeights'
fuzzy_labels_tag = 'fuzzyLabels'

# Regular expression to extract the number of an image
num_re_pattern = vol_tag + '_([0-9]+)_([0-9]+)' if petCtMode else vol_tag + '-([0-9]*)'
if lungMode:
        num_re_pattern += '.*lung([0-9]+)'
num_re = re.compile(num_re_pattern)

# Get the filenames of all the CT scans
image_names = listdir(path_nii)
vol_names = [name for name in image_names if vol_tag in name]

num_written = 0
for name_vol in vol_names:

        # Get the segmentation path and load the segmentation
        name_seg = name_vol.replace(vol_tag, seg_tag)   
        path_seg = join(path_nii, name_seg)
        if not isfile(path_seg):
            print("Warning: could not find segmentation " + path_seg)
            continue
        seg = nib.load(path_seg).get_data().astype(np.int64)

        # Get the file paths for the extras files
        extras_tags = {'voronoi': voronoi_tag,
                       'fuzzy_weights': fuzzy_weights_tag,
                       'fuzzy_labels': fuzzy_labels_tag
                      }
        extras_dtypes = {'voronoi': np.int32,
                        'fuzzy_weights': np.float32,
                        'fuzzy_labels': np.int32}
        extras_paths = {key: join(path_nii, name_vol.replace(vol_tag, tag)) for key, tag in extras_tags.items()}

        # Load the volume
        path_vol = join(path_nii, name_vol)
        if petCtMode:
            path_pet = path_vol
            path_ct = path_pet.replace('pet_', 'ct_')
            pet = nib.load(path_pet).get_data().astype(np.float32)
            ct = nib.load(path_ct).get_data().astype(np.float32)
            vol = np.concatenate((pet[:, :, :, np.newaxis], ct[:, :, :, np.newaxis]), axis=-1)
        else:
            vol = nib.load(path_vol).get_data().astype(np.float32)

        # Get the image number and training split
        matches = num_re.search(name_vol)
        if petCtMode:
            patientId = int(matches.group(1))
            studyId = int(matches.group(2))
            split = 'validation' if patientId in validationSet else 'training'
            img_name = '%d_%d' % (patientId, studyId)
            maxGroup = 2
        else:
            num_img = int(matches.group(1))
            split = 'validation' if num_img in validationSet else 'training'
            img_name = str(num_img)
            maxGroup = 1
        if lungMode:
            lungId = int(matches.group(maxGroup + 1))
            img_name += '_%d' % lungId
        print(img_name)
              
        # Write the h5 file
        path_save = join(path_h5[split], "%s%s.h5" % (img_name, tag))
        with h5py.File(path_save, 'w') as h5f:
            h5f.create_dataset('data', data=vol, dtype=np.float32)
            print('data')
            h5f.create_dataset('seg', data=seg, dtype=np.int64)
            print('seg')

            # Put the extras in the dataset
            for key, path in extras_paths.items():
                if not isfile(path):
                    continue
                dtype = extras_dtypes[key]
                vol = nib.load(path).get_data().astype(dtype)
                print(path)
                print(vol.shape)
                h5f.create_dataset(key, data=vol, dtype=dtype)
                print(key)

        print(path_save)
        num_written += 1
print('Done!')
