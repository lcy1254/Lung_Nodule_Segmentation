import pylidc as pl
import numpy as np
import pandas as pd
import os 
import scipy.ndimage
import matplotlib.pyplot as plt 
import pydicom
import cv2
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure
from skimage.transform import resize
from skimage import img_as_bool
from pylidc.utils import consensus
import nibabel as nib

def preprocess(scan):
    pre_image = scan.to_volume()
    
    #change the background to air 
    pre_image[pre_image <= -2000] = -1000
    
    #resample voxel size to 1mm^3
    resize = np.array([scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing])
    resized_shape = pre_image.shape * resize 
    new_shape = np.round(resized_shape,decimals=0)
    resize_factor = new_shape / pre_image.shape
    resampled_image = scipy.ndimage.zoom(pre_image, resize_factor)
    
    return resampled_image

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):

    binary_image = np.array(image > -700, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    temp = []
    temp.extend(np.unique(labels[:,:2,1:image.shape[2]:50]))
    temp.extend(np.unique(labels[:,-2:,1:image.shape[2]:50]))
    temp.extend(np.unique(labels[:5,:,1:image.shape[2]:50]))
    temp.extend(np.unique(labels[-5:,:,1:image.shape[2]:50]))
    background_label = np.unique(temp)
    
    for label in background_label: 
        if label != 2:
            binary_image[labels == label] = 2
        elif label == 2:
            None 
   
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, slice in enumerate(binary_image):
            slice = slice - 1
            labeling = measure.label(slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    
    # Dilation 
    binary_image = np.array(binary_image, dtype = np.uint8)
    kernel = np.ones((10,10), np.uint8)
    dilated = cv2.dilate(binary_image, kernel, iterations = 1)
 
    return dilated
# lung -- 1, everything else -- 0

def create_bounding_box(mask):
    indexes = np.argwhere(mask>0)
    x1 = min(indexes[:,0])
    x2 = max(indexes[:,0])
    y1 = min(indexes[:,1])
    y2 = max(indexes[:,1])
    z1 = min(indexes[:,2])
    z2 = max(indexes[:,2])
    return x1, x2, y1, y2, z1, z2

#Use consensus function 
def get_anns_con(scan):
    vol = scan.to_volume()
    nods = scan.cluster_annotations()
    
    anns_mask = []
    anns_bbox = []
    for i, nod in enumerate(nods):
        temp_anns_mask, temp_bbox, nmasks = consensus(nod)
        anns_mask.append(np.multiply(temp_anns_mask, 1))
        anns_bbox.append(temp_bbox)
        
    mask = np.zeros(vol.shape)
    for i in range(len(nods)):
        mask[anns_bbox[i]] = (anns_mask[i])
    return mask

def reshape_mask(mask, image):
    #resample voxel size to 1mm^3
    reshaped_mask = np.multiply(img_as_bool(resize(mask, image.shape)), 1)
    return reshaped_mask

def crop(image, mask, x1, x2, y1, y2, z1, z2):
    new_image = image[x1:x2, y1:y2, z1:z2]
    new_mask = mask[x1:x2, y1:y2, z1:z2]
    return new_image, new_mask 



#call only this one function to preprocess dicom files 
def full_preprocess(pid):
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == 'LIDC-IDRI-{}'.format(str(pid).zfill(4))).first()
    image = preprocess(scan) #reshaped
    lung_mask = segment_lung_mask(image, True) #reshaped
    x1, x2, y1, y2, z1, z2 = create_bounding_box(lung_mask) #reshaped
    unreshaped_mask = get_anns_con(scan) #not reshaped
    mask = reshape_mask(unreshaped_mask, image) #reshaped
    final_image, final_mask = crop(image, mask, x1, x2, y1, y2, z1, z2)   #final image and mask

    #creating nifti files (insert path)
    reshaped_final_image = final_image.reshape((final_image.shape[0],final_image.shape[1],final_image.shape[2],1))
    img = nib.Nifti1Image(reshaped_final_image, None)
    nib.save(img, os.path.join('D:\\lung_nifti_files','volume-{}.nii.gz'.format(pid)))   #edit path 
    labels = nib.Nifti1Image(final_mask, None)
    nib.save(labels, os.path.join('D:\\lung_nifti_files','labels-{}.nii.gz'.format(pid)))    #edit path 

'''
#sample code calling full_preprocess function
scans = pl.query(pl.Scan)
num = scans.count()
for i in range(75,num):
    try:
        full_preprocess(i+1)
    except:
        print(i+1)
'''
