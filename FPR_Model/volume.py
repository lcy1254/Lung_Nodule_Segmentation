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
        
        aug = {'rotate':None, 'shift':None, 'flip':None}
        choices = [True, False]
        
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

            aug['rotate'] = random.choice(choices)
            aug['shift'] = random.choice(choices)
            aug['flip'] = random.choice(choices)
            
            # Data Augmentation
            if aug['rotate'] == True: tempX = rotate(tempX)
            if aug['shift'] == True: tempX = shift(tempX)
            if aug['flip'] == True: tempX = flip(tempX)

            #resize to 50*50*50 with zero padding after augmentation
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
    vol = np.copy(volume)
    # define some rotation angles
    angles = [-20, -10, -5, 5, 10, 20]
    axis = [(1,0), (0,1), (1,2), (2,1), (2,0), (0,2)]
    # pick angles at random
    angle = random.choice(angles)
    axes = random.choice(axis)
    # rotate volume
    vol = ndimage.rotate(vol, angle, axes, reshape=False)
    return vol

def shift(volume):
    '''shift image'''
    vol = np.copy(volume)
    #define some shift values
    shift = []
    shifts = [-3, -2, -1, 0, 1, 2, 3]
    shift.append(random.choice(shifts))
    shift.append(random.choice(shifts))
    shift.append(random.choice(shifts))
    shift.append(0)
    vol = ndimage.shift(vol, shift)
    return vol
    
def flip(volume):
    '''flip image'''
    vol = np.copy(volume)
    flips = [(0),(0,1),(0,2),(0,1,2),(2),(2,1),(1)]
    flip = random.choice(flips)
    vol = np.flip(vol, flip)
    return vol
