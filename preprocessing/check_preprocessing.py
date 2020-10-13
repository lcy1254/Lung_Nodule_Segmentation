#those with problems (LIDC): 110, 181, 258, 311, 335, 395, 403, 467, 574, 814
#(pylidc) scans.count() says there's 1018 total scans but there's only 1012 patient ids 

import nibabel as nib 
import matplotlib.pyplot as plt

pid = 3   #enter patient ID
slice = 100    #enter slice # of scan you want to visualize 

img = nib.load('D:\\nifti_files\\lung_nifti_files\\volume-{}.nii.gz'.format(pid))   #change path 
print(img.dataobj.shape)
plt.imshow(img.dataobj[:,:,slice,0])
labels = nib.load('D:\\nifti_files\\lung_nifti_files\\labels-{}.nii.gz'.format(pid))    #change path 
print(labels.dataobj.shape)
plt.imshow(labels.dataobj[:,:,slice])
