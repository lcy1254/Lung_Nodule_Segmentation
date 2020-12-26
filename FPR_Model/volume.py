'''
load data & preprocess & perform augmentation

adapted from https://github.com/shervinea/enzynet and https://keras.io/examples/vision/3D_image_classification/
'''

import keras
import numpy as np
from scipy import ndimage

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, listIDs, dataDir, batch_size=32, v_size=50, n_channels=1, n_classes=2, shuffle=True):
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
        
        for i, ID in enumerate(temp_listIDs):
            # Load data
            tempX = read_h5_file(os.path.join(self.dataDir, '{}.h5'.format(ID)))
            tempy = read_txt_file(os.path.join(self.dataDir, '{}.txt'.format(ID)))
            
            # Preprocessing
            tempX = normalize(tempX)
            tempX = resize_volume(tempX, self.v_size)
            
            # Data Augmentation
            tempX = rotate(tempX)
            #tempX = shift(tempX)
            tempX = flip(tempX)
            
            # Add another dim
            tempX = np.expand_dims(tempX, axis=3)
            
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
    
    curDepth = img.shape[-1]
    curWidth = img.shape[0]
    curHeight = img.shape[1]
    
    if curDepth<=sideLength and curWidth<=sideLength and curHeight<=sideLength:
        vol = np.ones((sideLength,sideLength,sideLength),dtype='float32')*-1000
        vol[:curWidth,:curHeight,:curDepth] = img
    else:
        print('img size bigger than {}'.format(sideLength))
        vol = img[:sideLength, :sideLength, :sideLength]
    return vol

def rotate(volume):
    """Rotate the volume by a few degrees"""
    vol = np.copy(volume)
    # define some rotation angles
    angles = [-20, -10, -5, 0, 5, 10, 20]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    vol = ndimage.rotate(vol, angle, reshape=False, cval=-1000)
    return vol

def shift(volume):
    #don't use yet -- image is located at corner
    '''shift image'''
    vol = np.copy(volume)
    #define some shift values
    shifts = [5, 10, 15]    #edit these values later
    shift = random.choice(shifts)
    vol = ndimage.shift(vol, shift, cval=-1000)
    return vol
    
def flip(volume):
    '''flip image'''
    vol = np.copy(volume)
    flips = [(0,0),(0),(0,1),(0,2),(0,1,2),(2),(2,1),(1)]
    flip = random.choice(flips)
    vol = np.flip(vol, flip)
    return vol
