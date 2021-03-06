'''
load data & preprocess & perform augmentation

adapted from https://github.com/shervinea/enzynet and https://keras.io/examples/vision/3D_image_classification/
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy import ndimage
import os
import h5py
import random
'''
LJDFLJSDLFJLSDJFLSDLGJLSD
SDLJFSDJFLSJKDFLJSDLFJK
SDJFLSJDLFJSLDFJSLDJFLSD
ENABLE ROTATE BEFORE YOU TRAIN ANY OTHER MODEL

CHANGE SHIFT VALUES DEPENDING ON 96 or 48 
'''
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, listIDs, dataDir, v_size, batch_size=32, n_channels=1, n_classes=2, shuffle=True):
        #adjust v_size as necessary !!
        'Initialization'
        self.v_size = v_size
        self.batch_size = batch_size
        self.listIDs = listIDs
        self.dataDir = dataDir
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.listIDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, temp_listIDs):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        #Initialization
        X = np.empty((self.batch_size, self.v_size, self.v_size, self.v_size, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        #aug = {'rx':None, 'ry':None, 'rz':None, 'shift':None, 'flip':None}
        #choices = [True, False]
        
        for i, ID in enumerate(temp_listIDs):
            # Load data
            tempX = read_h5_file(os.path.join(self.dataDir, '{}.h5'.format(ID)))
            if tempX.shape[3] != 1:
                print("{} does not have channel at end".format(ID))
                a = list(tempX.shape)
                a.remove(1)
                tempX = tempX.reshape(a[0], a[1], a[2], 1)
            tempy = read_txt_file(os.path.join(self.dataDir, '{}.txt'.format(ID)))
            
            # Preprocessing
            tempX = normalize(tempX)
            
            #aug['rx'] = random.choice(angles)
            #aug['ry'] = random.choice(angles)
            #aug['rz'] = random.choice(angles)
            #aug['shift'] = random.choice(shifts)
            #aug['flip'] = random.choice(flips)
            
            # Data Augmentation
            #if aug['rotate'] == True: tempX = rotate(tempX)
            #if aug['shift'] == True: tempX = shift(tempX)
            #if aug['flip'] == True: tempX = flip(tempX)
    
            xform = list(get_xform(tempX, rotMax=(90, 90, 90), pReflect=(.5, .5, .5), shearMax=(.5,.5,.5), transMax=(3,3,3)))
            tempX = apply_xforms_images(xform, list(tempX))
            
            #resize to 48*48*48 with zero padding after augmentation
            if tempX.shape[0] == self.v_size and tempX.shape[1] == self.v_size and tempX.shape[2] == self.v_size and tempX.shape[3] == 1:
                pass
            else:
                tempX = resize_volume(tempX, self.v_size)
            
            # Store sample
            X[i,] = tempX
            # Store class
            y[i] = tempy
      
        return X, y
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.listIDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        #Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #Find list of IDs
        temp_listIDs = [self.listIDs[i] for i in indexes]
        
        #Generate data
        X, y = self.__data_generation(temp_listIDs)
        
        return X, y

class testDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, listIDs, dataDir, v_size, batch_size=32, n_channels=1):
        #adjust v_size as necessary !!
        'Initialization'
        self.v_size = v_size
        self.batch_size = batch_size
        self.listIDs = listIDs
        self.dataDir = dataDir
        self.indexes = np.arange(len(self.listIDs))
        self.n_channels = n_channels
            
    def __data_generation(self, temp_listIDs):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        #Initialization
        X = np.empty((self.batch_size, self.v_size, self.v_size, self.v_size, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        for i, ID in enumerate(temp_listIDs):
            # Load data
            tempX = read_h5_file(os.path.join(self.dataDir, '{}.h5'.format(ID)))
            tempy = read_txt_file(os.path.join(self.dataDir, '{}.txt'.format(ID)))
            
            # Preprocessing
            tempX = normalize(tempX)
            
            #resize to 50*50*50 with zero padding after augmentation
            tempX = resize_volume(tempX, self.v_size)
            
            # Store sample
            X[i,] = tempX
            # Store class
            y[i] = tempy
            
            #print('index: {}// ID: {}// true class: {}'.format(i, ID, tempy))
      
        return X, y
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.listIDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        #Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #Find list of IDs
        temp_listIDs = [self.listIDs[i] for i in indexes]
        
        #Generate data
        X, y = self.__data_generation(temp_listIDs)
        
        return X, y

def read_h5_file(filepath):
    '''read and load volume'''
    hf = h5py.File(filepath, 'r')
    scan = np.array(hf.get('data'))
    return scan

def read_txt_file(filepath):
    '''read and load class'''
    with open(filepath, 'r') as f:
        label = int(f.readline().strip())
    return label

def normalize(volume):
    '''normalize the volume'''
    min = -1000
    max = 400
    volume[volume<min] = min
    volume[volume>max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img, sideLength):
    sideLength = sideLength #change later
    
    curDepth = img.shape[2]
    curWidth = img.shape[0]
    curHeight = img.shape[1]
    
    if curDepth>sideLength: curDepth=sideLength
    if curWidth>sideLength: curWidth=sideLength
    if curHeight>sideLength: curHeight=sideLength
    
    vol = np.zeros((sideLength,sideLength,sideLength,1),dtype='float32')
    halfWidth = int(curWidth/2)
    halfHeight = int(curHeight/2)
    halfDepth = int(curDepth/2)
    startWidth = int((sideLength/2)-halfWidth)
    startHeight = int((sideLength/2)-halfHeight)
    startDepth = int((sideLength/2)-halfDepth)
    if startWidth<0: startWidth = 0
    if startHeight<0: startHeight = 0
    if startDepth<0: startDepth = 0
    endWidth = int(startWidth+curWidth)
    endHeight = int(startHeight+curHeight)
    endDepth = int(startDepth+curDepth)
    vol[startWidth:endWidth,startHeight:endHeight,startDepth:endDepth,:] = img[:curWidth,:curHeight,:curDepth,:]
    
    return vol

def rotate(volume):
    """Rotate the volume by a few degrees"""
    # define some rotation angles
    angles = [-20, -10, -5, 5, 10, 20]
    axis = [(1,0), (0,1), (1,2), (2,1), (2,0), (0,2)]
    # pick angles at random
    angle = random.choice(angles)
    axes = random.choice(axis)
    # rotate volume
    vol = ndimage.rotate(volume, angle, axes, reshape=False)
    return vol

def shift(volume):
    '''shift image'''
    #define some shift values
    shift = []
    shifts = [-3, -2, -1, 0, 1, 2, 3]
    shift.append(random.choice(shifts))
    shift.append(random.choice(shifts))
    shift.append(random.choice(shifts))
    shift.append(0)
    vol = ndimage.shift(volume, shift)
    return vol
    
def flip(volume):
    '''flip image'''
    flips = [(0),(0,1),(0,2),(0,1,2),(2),(2,1),(1)]
    flip = random.choice(flips)
    vol = np.flip(volume, flip)
    return vol


import math
import numpy as np
import scipy.ndimage as nd

from pyCudaImageWarp import cudaImageWarp

"""
    Pad the image to have a singleton channel dimension.
"""
def __pad_channel__(im):
    ndim = 3
    return np.expand_dims(im, ndim) if len(im.shape) < ndim + 1 else im

"""
    Adjust the translation component of an affine transform so that it maps
    'point' to 'target'. Does not change the linear component.

"""
def set_point_target_affine(mat, point, target):
    mat = mat.astype(float)
    mat[0:3, 3] = target - mat[0:3, 0:3].dot(point[np.newaxis].T).T
    return mat

def jitter_mask(labels, pQuit=0.5, maxIter=1, pKeep=0.5, pJagged=0.5):
    """
        Slightly modify a set of labels, with randomness. Only modifies the
        image mask, that is, the labels less than zero. Jitters the perimeter
    """
    # With probability pQuit, do nothing at all
    if np.random.uniform() <= pQuit:
        return labels

    # Do nothing if all the labels are valid
    invalid = labels == -1
    if not np.any(invalid):
        return labels

    # Randomly draw the number of iterations
    iters = int(round(np.random.uniform(low=1, high=maxIter)))

    # Erode or dilate smoothly
    if np.random.uniform() > pJagged:
        if np.random.uniform() > 0.5:
            invalid = nd.morphology.binary_erosion(invalid, iterations=iters)
        else:
            invalid = nd.morphology.binary_dilation(invalid, iterations=iters)
    else:
        # Jitter the boundary in each iteration
        for i in range(iters):

            # Chose whether to erode or dilate
            if np.random.uniform() > 0.5:
                new = nd.morphology.binary_erosion(invalid)
            else:
                new = nd.morphology.binary_dilation(invalid)

            # Get the difference and randomly choose whether to keep them
            diff = new ^ invalid
            invalid[diff] = np.random.uniform(size=(np.sum(diff),)) <= pKeep

    # Return the result
    result = np.zeros_like(labels)
    result[invalid] = -1
    result[~invalid] = labels[~invalid]
    return result

def get_translation_affine(offset):
    """
        Returns a 4x4 affine matrix (homogeneous coordinates) shifting by the
        given offset.
    """
    mat = np.eye(4)
    mat[0:3, 3] = offset
    return mat

"""
    Check that the image shape is compatible with the xform shape, up to ndim.
    Ignores channels unless they're >1.
"""
def __check_shapes__(imShape, xformShape, ndim=3):
    hasChannels = len(imShape) > ndim and imShape[ndim] > 1
    if hasChannels and xformShape[ndim] != imShape[ndim]:
        raise ValueError("Output shape has %d channels, while input has %d" % \
                (xformShape[3], imShape[3]))
    if len(xformShape[:ndim]) != len(imShape[:ndim]):
        raise ValueError("""
                Input and output shapes have mismatched number of dimensions.
                Input: %s, Output: %s"
                """ % (xformShape, imShape))

        crop_offset = np.random.uniform(low=-crop_half_range, high=crop_half_range)
        crop_center = im_center + crop_offset

def __shape_center__(shape):
    return (np.array(shape[:3]) - 1.0) / 2.0

def __crop_uniform__(im_center, crop_half_range):
        crop_offset = np.random.uniform(low=-crop_half_range, high=crop_half_range)
        crop_center = im_center + crop_offset
        return crop_center

def __crop_in_mask__(crop_half_range, mask, printFun=None):
    """
        Crop only in this mask, if at all possible
    """

    # Compute shape parameters
    im_center = __shape_center__(mask.shape)

    # Check if the mask is empty
    if not np.any(mask):
        if printFun is not None:
            printFun("Defaulting to uniform crop...")
        return __crop_uniform__(im_center, crop_half_range)

    # Pick a random center in the range
    center_idx = np.random.choice(np.nonzero(mask.flatten())[0])
    crop_center = np.array(np.unravel_index(center_idx, mask.shape))

    # Pick the nearest valid crop to this center
    crop_disp = crop_center - im_center
    crop_disp_valid = np.minimum(crop_half_range,
        np.maximum(crop_disp, -crop_half_range)
    )
    crop_center = im_center + crop_disp_valid

    return crop_center
"""
    Randomly generates a 3D affine map based on the given parameters. Then
    applies the map to warp the input image and, optionally, the segmentation.
    Warping is done on the GPU using pyCudaImageWarp. By default, the output
    shape is the same as that of the input image.

    By default, the function only generates the identity map. The affine
    transform distribution is controlled by the following parameters:
        im - The input image, a numpy array.
        seg - The input segmentation, same shape as im (optional).
        shape - The output shape (optional).
        init - The initial linear transform. Defaults to identity.
        rotMax - Uniform rotation about (x,y,z) axes. For example, (10,10,10)
            means +-10 degrees in about each axis.
        pReflect - Chance of reflecting about (x,y,z) axis. For example,
                (.5, 0, 0) means there is a 50% chance of reflecting about the
                x-axis.
        shearMax - Uniform shearing about each axis. For example, (1.1, 1.1,
                1.1) shears in each axis in the range (1.1, 1 / 1.1)
        transMax - Uniform translation in each coordinate. For example, (10, 10,
                10) translates by at most +-10 voxels in each coordinate.
        otherScale - Gaussian-distributed affine transform. This controls the
                variance of each parameter.
        randomCrop - Choose whether to randomly crop the image. Possible modes:
            'none' - Do no cropping (default).
            'uniform' - All crops are equally likely.
            'valid' - Like uniform, but only for crops with non-negative label.
            'nonzero' - Choose only from crops whose centers have a positive
                label. Cannot be used if segList is None.
        noiseLevel - An array of C elements. Decide the amount of noise for each channel
            using this standard deviation.
        windowMin - A 2xC matrix, where C is the number of channels in im,
            from which the lower window threshold is sampled uniformly. By
            default, this does nothing. The cth row defines the limits for the
            cth channel.
        windowMax - A matrix from which the upper window threshold is
            sampled uniformly. Same format as winMin. By default, this does
            nothing.
        occludeProb - Probability that we randomly take out a chunk of out of
            the image.
        oob_label - The label assigned to out-of-bounds pixels (default: 0)
        printFun - If provided, use this function to print the parameters.
        oob_image_val - If provided, set out-of-bounds voxels to this value.
        api - The underlying computation platform. Either 'cuda' or 'scipy'.
        device - The index of the CUDA device, if provided.

    All transforms fix the center of the image, except for translation.
"""
def get_xform(im, seg=None, shape=None, rand_seed=None,
    rotMax=(0, 0, 0), pReflect=(0, 0, 0), init=np.eye(3),
    shearMax=(1,1,1), transMax=(0,0,0), otherScale=0, randomCrop='none',
    noiseLevel=None, windowMin=None, windowMax=None,
    occludeProb=0.0, printFun=None):

    # Default to have the same output and input shape
    if shape is None:
        shape = im.shape

    # Pad the image to have a channel dimension
    ndim = 3
    im = __pad_channel__(im)

    # Pad the shape with missing dimensions
    if len(shape) < ndim + 1:
        shape = shape + (1,) * (ndim + 1 - len(shape))
    numChannels = shape[-1]

    # Check that the input and output shapes are compatible
    __check_shapes__(im.shape, shape)

    #  Set the random seed, if specified
    if rand_seed is not None:
        np.random.seed(rand_seed)

    # ---Randomly generate the desired transforms, in homogeneous coordinates---
    
    # Draw the noise level
    if noiseLevel is not None:
        noiseScale = [np.abs(np.random.normal(scale=n)) \
            if n > 0 else 0 for n in noiseLevel]
    else:
        noiseScale = np.zeros(im.shape[-1])

    # Draw the width of occlusion, if any
    if np.random.uniform() < occludeProb:
        occludeWidth = int(np.floor(np.random.uniform(low=0,
                high=im.shape[2] / 2)))
    else:
        occludeWidth = None

    mat_init = np.identity(4)
    mat_init[0:3, 0:3] = init

    # Get the center of the input volume
    im_center = __shape_center__(im.shape)
    shape_center = __shape_center__(shape)

    # Compute the input crop center
    if printFun is not None:
        printFun("cropType: %s" % randomCrop)
    crop_half_range = np.maximum(im_center - shape_center, 0)
    if randomCrop == 'none':
        crop_center = im_center
    elif randomCrop == 'uniform':
        crop_center = __crop_uniform__(im_center, crop_half_range)
    elif randomCrop == 'valid':
        if seg is None:
            raise ValueError('Cannot use randomCrop == \'valid\' when seg is not provided!')
        # Take the intersection of the crop range and valid classes
        crop_center = __crop_in_mask__(crop_half_range, seg >= 0)
    elif randomCrop == 'nonzero':
        if seg is None:
            raise ValueError('Cannot use randomCrop == \'nonzero\' when seg is not provided!')
        # First pick a class, accounting for label shifting
        classes = np.unique(seg.flatten())
        classes = classes[classes >= 1]
        if len(classes) == 0:
            if printFun is not None:
                printFun("Defaulting to uniform crop...")
            crop_center = __crop_uniform__(im_center, crop_half_range)
        else:
            rand_class = np.random.choice(classes)

            if printFun is not None:
                printFun("crop_class: %d" % rand_class)

            # Same as valid crop, but with this class
            crop_center = __crop_in_mask__(crop_half_range, seg == rand_class)

    else:
        raise ValueError('Unrecognized randomCrop: ' + randomCrop)

    # Uniform rotation
    rotate_deg = np.random.uniform(low=-np.array(rotMax), high=rotMax)
    lin_rotate = np.identity(3)
    for i in range(3): # Rotate about each axis and combine
        # Compute the angle of rotation, in radians
        rad = rotate_deg[i] * 2 * math.pi / 360

        # Form the rotation matrix about this axis
        rot = np.identity(3)
        axes = [x for x in range(3) if x != i]
        rot[axes[0], axes[0]] = math.cos(rad)
        rot[axes[0], axes[1]] = -math.sin(rad)
        rot[axes[1], axes[0]] = -rot[axes[0], axes[1]]
        rot[axes[1], axes[1]] = rot[axes[0], axes[0]]

        # Compose all the rotations
        lin_rotate = lin_rotate.dot(rot)

    # Extend the linear rotation to an affine transform
    mat_rotate = np.identity(4)
    mat_rotate[0:3, 0:3] = lin_rotate

    # Uniform shear, same chance of shrinking and growing
    shearMax = np.array(shearMax)
    if np.any(shearMax <= 0):
        raise ValueError("Invalid shearMax: %f" % (shear))
    #shear = np.random.uniform(low=1.0, high=shearMax, size=3)
    shearScale = np.abs(shearMax - 1.0)
    shear = np.array([np.random.normal(loc=1.0,
        scale=float(s) / 4) if s > 0 else 1.0 for s in shearScale])
    invert_shear = np.random.uniform(size=3) < 0.5
    shear[invert_shear] = [1.0 / s if s != 0 else 0 for s in shear[invert_shear]]
    mat_shear = np.diag(np.hstack((shear, 1)))

    # Reflection
    do_reflect = np.random.uniform(size=3) < pReflect
    mat_reflect = np.diag(np.hstack((1 - 2 * do_reflect, 1)))

    # Generic affine transform, Gaussian-distributed
    mat_other = np.identity(4)
    mat_other[0:3, :] = mat_other[0:3, :] + \
        (np.random.normal(loc=0.0, scale=otherScale, size=(3,4)) \
            if otherScale > 0 else 0)

    # Uniform translation
    transMax = np.array(transMax)
    translation = np.random.uniform(low=-transMax,
            high=transMax) if np.any(transMax > 0) else np.zeros_like(transMax)

    # Compose all the transforms, fix the center of the crop
    mat_total = set_point_target_affine(
        mat_rotate.dot( mat_shear.dot( mat_reflect.dot( mat_other.dot( mat_init)
        ))),
        shape_center,
        crop_center + translation
    )

    # Any columns with infinity  are unchanged
    winMin = np.array([-float('inf') for x in range(numChannels)])
    winMax = np.array([float('inf') for x in range(numChannels)])
    validCols = ~np.any(
        (windowMin is not None and (np.abs(windowMin) == float('inf'))
            | (windowMax is not None and np.abs(windowMax) == float('inf'))),
        axis=0
    )

    # Draw the window thresholds uniformly in the specified range
    numChannels = shape[-1]
    if windowMin is not None:
        winMin[validCols] = np.random.uniform(
            low=windowMin[0, validCols],
            high=windowMin[1, validCols]
        )
    if windowMax is not None:
        winMax[validCols] = np.random.uniform(
            low=windowMax[0, validCols],
            high=windowMax[1, validCols]
        )

    # Draw the occlusion parameters
    if occludeWidth is not None:
        # Take a chunk out at random
        occZmin = int(np.floor(np.random.uniform(
                low=-occludeWidth, high=im.shape[2])))
        occZmax = occZmin + occludeWidth - 1
    else:
        # By default, do no occlusion
        occZmin = 0
        occZmax = -1

    # Optionally print the result
    if printFun is not None:
        printFun("crop_center: [%d, %d, %d]" % (crop_center[0], crop_center[1], crop_center[2]))
        printFun("occZmin: %d occZmax: %d" % (occZmin, occZmax))
        printFun("winmin: %s winmax: %s" % (winMin, winMax))
        printFun("rotation: [%d, %d, %d]" % (rotate_deg[0], rotate_deg[1],
                rotate_deg[2]))
        printFun("translation: [%d, %d, %d]" % (translation[0], translation[1],
                translation[2]))
    # Return a dict containing all the transform parameters
    return {
        'affine': mat_total,
        'occZmin': occZmin,
        'occZmax': occZmax,
        'winMin': winMin,
        'winMax': winMax,
        'noiseScale': noiseScale,
        'shape': shape
    }

"""
    Choose the implementation based on the api string.
"""
def __get_pushFun_popFun__(api):
    if api == 'cuda':
        pushFun = cudaImageWarp.push
        popFun = cudaImageWarp.pop
    elif api == 'scipy':
        from pyCudaImageWarp import scipyImageWarp
        pushFun = scipyImageWarp.push
        popFun = scipyImageWarp.pop
    else:
        raise ValueError('Unrecognized api: ' + api)

    return pushFun, popFun

def apply_xforms_images(xformList, imList, oob=0,
        api='cuda', device=None):
    """
        Shortcut for only images.
    """
    return apply_xforms(xformList, imList=imList, oob_image=oob,
        api=api, device=device)

def apply_xforms_labels(xformList, labelsList, oob=0,
        api='cuda', device=None):
    """
        Shortcut for only labels.
    """
    return apply_xforms(xformList, labelsList=labelsList, oob_labels=oob,
        api=api, device=device)

def apply_xforms(xformList, imList=None, labelsList=None, oob_image=0,
        oob_label=0, api='cuda', device=None):
    """
        Apply transforms which were created with get_xform.
    """

    # Check the operating mode
    haveImages = imList is not None
    haveLabels = labelsList is not None
    if not haveImages and not haveLabels:
        raise ValueError("Received neither images nor labels!")

    # Verify inputs
    if haveImages and len(xformList) != len(imList):
        raise ValueError("Received %d xforms but %d images" % (len(xformList),
            len(imList)))
    if haveLabels and len(xformList) != len(labelsList):
        raise ValueError("Received %d xforms but %d labels" % (len(xformList),
            len(labelsList)))

    # Get the implementation
    pushFun, popFun = __get_pushFun_popFun__(api)

    # Push all the images
    if haveImages:
        __push_xforms_images__(pushFun, xformList, imList, oob_image, device)

    # Push all the labels
    if haveLabels:
        __push_xforms_labels__(pushFun, xformList, labelsList, oob_label,
                device)

    # Pop all the images
    returns = []
    if haveImages:
        returns.append(__pop_xforms__(imList, xformList, popFun))

    # Pop all the labels
    if haveLabels:
        returns.append(__pop_xforms__(labelsList, xformList, popFun))

    return tuple(returns)

def __pop_xforms__(imList, xformList, popFun):
    """
        Shortcut to pop a list of outputs, from the given inputs and xforms.
    """
    augImList = []
    for im, xform in zip(imList, xformList):
        shape = xform['shape'][:len(im.shape)]
        augImList.append(__pop_xform(shape, im.dtype, popFun))

    return augImList

def __push_xforms_images__(*args):
    """
        Push a list of images. Arguments same as __push_xforms__, except
        supplies pushTypeFun.
    """
    __push_xforms__(__push_xform_image__, *args)

def __push_xforms_labels__(*args):
    """
        Push a list of labels. Arugments same as __push_xforms_images__.
    """
    __push_xforms__(__push_xform_labels__, *args)

def __push_xforms__(pushTypeFun, pushFun, xformList, imList, oob, device):
    """
        Shortcut to push a list of images or labels, using pushTypeFun. Not
        called directly.
    """
    for im, xform in zip(imList, xformList):
        pushTypeFun(xform, im, pushFun, oob, device)

def __push_xform_image__(xform, im, pushFun, oob, device):
    """
        Start processing an image. Called by apply_xforms.
    """

    # Add a channel dimension
    im = __pad_channel__(im)

    # Check the shapes
    __check_shapes__(im.shape, xform['shape'])

    # Warp each image channel the same way
    warp_affine = xform['affine'][0:3, :]
    shape = xform['shape'][:3]
    numChannels = xform['shape'][3]
    for c in range(numChannels):
        pushFun(
                im[:, :, :, c],
                warp_affine,
                interp='linear',
                shape=shape,
                std=xform['noiseScale'][c],
                winMin=xform['winMin'][c],
                winMax=xform['winMax'][c],
                occZmin=xform['occZmin'],
                occZmax=xform['occZmax'],
                oob=oob,
                device=device
        )

def __push_xform_labels__(xform, labels, pushFun, oob, device):
    """
        Like __push_xform_image__, but for labels.
    """

    # Check the shapes
    __check_shapes__(labels.shape[:3], xform['shape'][:3])

    warp_affine = xform['affine'][0:3, :]
    shape = xform['shape'][:3]
    pushFun(
        labels,
        warp_affine,
        interp='nearest',
        shape=shape,
        occZmin=xform['occZmin'],
        occZmax=xform['occZmax'],
        oob=oob,
        device=device
    )

    return

def __pop_xform(shape, dtype, popFun):
    """
    Finish processing an image, and return the result. Squeezes out the channel
    dimension, if necessary.
    """

    # Pop multi-channel images one channel at a time
    if len(shape) > 3 and shape[3] > 1:
        im = np.zeros(shape, dtype=dtype, order='F')
        for c in range(shape[3]):
            im[:, :, :, c] = popFun()
        return im

    # Pop a single-channel image
    return popFun()
