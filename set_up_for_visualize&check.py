      #import necessary modules 
import sys
np.set_printoptions(threshold=sys.maxsize)
import nibabel as nib 
import matplotlib.pyplot as plt
import numpy as np

      #get image and labels & set iter, volume # value 
iter = 14701    #manually set
vol = 29        #manually set 

img = nib.load('/Volumes/Transcend/nifti_files/lung_nifti_files/labels-{}.nii.gz'.format(vol))
print(img.dataobj.shape)
labels = nib.load('/Users/ChaeyoungLee/Downloads/inference/iter_{}/prediction/inferred_volume_{}_model_iter_{}.nii.gz'.format(iter, vol, iter))
print(labels.dataobj.shape)

      #visualize gt or pred for one slice value
slice = 46      #manually set 
fig, ax1 = plt.subplots(1, figsize = (15,15))
ax1.imshow(labels.dataobj[:,:,slice])

      #visualize gt and pred for one slice value
slice = 209     #manually set 
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (20,20))
ax1.set_title('Ground Truth')
ax1.imshow(img.dataobj[:,:,slice])
ax2.set_title('Prediction')
ax2.imshow(labels.dataobj[:,:,slice])
plt.show()

      #visualize gt and pred for two slice values (produces 4 plots)
slice = 209     #manually set 
slice2 = 220    #manually set 
fig, (ax1,ax2) = plt.subplots(2,2, figsize = (20,20))
#ax1.set_title('Ground Truth')
ax1[0].set_title('GT slice')
ax1[0].imshow(img.dataobj[:,:,slice])
ax1[1].set_title('GT slice2')
ax1[1].imshow(img.dataobj[:,:,slice2])
#ax2.set_title('Prediction')
ax2[0].set_title('PRED slice')
ax2[0].imshow(labels.dataobj[:,:,slice])
ax2[1].set_title('PRED slice2')
ax2[1].imshow(labels.dataobj[:,:,slice2])
plt.show()

      #visualize nodules in 3D 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

a = labels.dataobj    #or something else like img.dataobj (manually set)
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
pos = np.where(np.array(a)==1)
ax.scatter(pos[0], pos[1], pos[2], c='black')
ax.set(xlim = (1, a.shape[0]), ylim = (1, a.shape[1]), zlim = (1, a.shape[2]))
plt.show()

      #print z values of nodules 
truth = np.array(img.dataobj)
q,w,e = np.where(truth == 1)
print("truth: ", np.unique(e))
arr = np.array(labels.dataobj)
a,b,c = np.where(arr == 1)
print("prediction: ", np.unique(c))

      #print accuracy of one nodule and confusion of prediction labels 
a = boundingBox(img.dataobj)
acc = trueNod_detection_accuracy(a[0], labels.dataobj, img.dataobj)  #manually set index of a --> which nodule 
print("%.6f"%acc)
confusion = compute_confusion(labels.dataobj, img.dataobj)
print(confusion)
