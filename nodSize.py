import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage.segmentation import find_boundaries
#import cv2
import math

def nodVol(nodArr):
    return np.sum(nodArr)

def nodSizeRoot(nodArr):
    #computes diameter with square root of largest cross-sectional area
    greatest_area = 0
    for i in range(nodArr.shape[0]):
        tempArea = np.sum(nodArr[i,:,:])
        if tempArea > greatest_area:
            greatest_area = tempArea
    return 2*math.sqrt(greatest_area/math.pi)
    
def nodDia(nodArr):
    #computes max diameter in axial plane in nodule
    #FIX: need to compute perpendicular maximal short axis measurement
    greatest_diameter = 0
    for i in range(nodArr.shape[0]):
        img = 1*nodArr[i,:,:]
        boun = find_boundaries(img, mode='inner').astype(np.uint8)
        if np.sum(boun) == 0: continue
        x, y = np.nonzero(boun)
        indices = np.column_stack((x, y))
        diameters = squareform(pdist(indices))
        diameter = diameters.max()

        if diameter > greatest_diameter:
            greatest_diameter = diameter
            '''
            x = i
            y,z = np.unravel_index(diameters.argmax(), diameters.shape)
            a1, b1 = indices[y]
            a2, b2 = indices[z]
    tempSlope = (b1-b2)/(a1-a2)
    perpSlope = -1/tempSlope

    for i in range(nodArr.shape[0]):
        indices, _= cv2.findContours(nodArr(i,:,:))
        diameters = squareform(pdist(indices))
    '''
    return greatest_diameter
    
def nodAbsDia(nodArr):
    #computes absolute max diameter in nodule (off-axis)
    x,y,z = np.nonzero(nodArr)
    indices = np.array((x,y,z)).transpose()
    diameters = squareform(pdist(indices))
    maxDia = diameters.max()
    return maxDia
