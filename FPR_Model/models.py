import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
    
def VGG16(sideLength):
    '''
    VGG16 3D CNN Implementation
    '''
    #try decreasing strides
    #compare with and without batch norm
    
    input = keras.Input(shape=(sideLength, sideLength, sideLength, 1))
    
    x = layers.ZeroPadding3D(padding=(1,1,1))(input)
    x = layers.Conv3D(64, 3, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding3D((1,1,1))(x)
    x = layers.Conv3D(64, 3, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((2,2,2), strides=(2,2,2))(x)
    
    x = layers.ZeroPadding3D((1,1,1))(x)
    x = layers.Conv3D(128, 3, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding3D((1,1,1))(x)
    x = layers.Conv3D(128, 3, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((2,2,2), strides=(2,2,2))(x)
    
    x = layers.ZeroPadding3D((1,1,1))(x)
    x = layers.Conv3D(256, 3, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding3D((1,1,1))(x)
    x = layers.Conv3D(256, 3, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding3D((1,1,1))(x)
    x = layers.Conv3D(256, 3, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((2,2,2), strides=(2,2,2))(x)
    
    '''
    x = layers.ZeroPadding3D((1,1,1))(x)
    x = layers.Conv3D(512, 3, 1, activation='relu')(x)
    x = layers.ZeroPadding3D((1,1,1))(x)
    x = layers.Conv3D(512, 3, 1, activation='relu')(x)
    x = layers.ZeroPadding3D((1,1,1))(x)
    x = layers.Conv3D(512, 3, 1, activation='relu')(x)
    x = layers.MaxPooling3D((2,2,2), strides=(2,2,2))(x)
    '''
    
    x = layers.ZeroPadding3D((1,1,1))(x)
    x = layers.Conv3D(512, 3, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding3D((1,1,1))(x)
    x = layers.Conv3D(512, 3, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding3D((1,1,1))(x)
    x = layers.Conv3D(512, 3, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((2,2,2), strides=(2,2,2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(input, output, name='VGG16')
    return model
    
def alexNet(sideLength):
    '''AlexNet 3D CNN Implementation'''
    input = keras.Input(shape=(sideLength, sideLength, sideLength, 1))
    
    x = layers.Conv3D(filters=96, kernel_size=11, strides=4, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((3,3,3), strides=2)(x)
    
    x = layers.Conv3D(256, 5, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(2, strides=1)(x)
    
    x = layers.Conv3D(384, 3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv3D(384, 3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv3D(256, 3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((3,3,3), (2,2,2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(input, output, name='AlexNet')
    return model
    
def resNet(sideLength):
    '''resNet 3D implementation'''
    input = keras.Input(shape=(sideLength, sideLength, sideLength, 1))
    
    x = layers.ZeroPadding3D(padding=(1,1,1))(input)
    x = layers.Conv3D(filters=64, kernel_size=7, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((3,3,3), strides=(2,2,2))(x)
    
    for i in range(3): x = residual_block(x, 64, 256)
    for i in range(4): x = residual_block(x, 128, 512)
    for i in range(6): x = residual_block(x, 256, 1024)
    for i in range(3): x = residual_block(x, 512, 2048)
    
    #at the end
    x = layers.GlobalAveragePooling3D((2,2,2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(input, output, name='resnet')
    return model

def residual_block(layer_in, f1N2, f3):
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != f3:
        merge_input = layers.Conv3D(f3, (1,1,1), strides=(1,1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
        
        #conv1
        conv1 = layers.Conv3D(f1N2, (1,1,1), strides=(2,2,2), padding='same', kernel_initializer='he_normal')(layer_in)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Activation('relu')(conv1)
    else:
        # conv1
        conv1 = layers.Conv3D(f1N2, (1,1,1), padding='same', kernel_initializer='he_normal')(layer_in)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Activation('relu')(conv1)
        
    # conv2
    conv2 = layers.Conv3D(f1N2, (3,3,3), padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    #conv3
    conv3 = layers.Conv3D(f3, (1,1,1), padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    # add filters, assumes filters/channels last
    layer_out = layers.Add()([conv3, merge_input])
    # activation function
    layer_out = layers.Activation('relu')(layer_out)
    return layer_out

def vgg_block(layer_in, n_filters, n_conv):
    # add convolutional layers
    for _ in range(n_conv):
        layer_in = layers.Conv3D(n_filters, (3,3,3), activation='relu')(layer_in)
    # add max pooling layer
    layer_in = layers.MaxPooling3D((2,2,2), strides=(2,2,2))(layer_in)
    return layer_in


def _shortcut3d(input, residual):
    """3D shortcut to match input and residual and merges them with "sum"."""
    stride_dim1 = ceil(input._keras_shape[DIM1_AXIS] \
        / residual._keras_shape[DIM1_AXIS])
    stride_dim2 = ceil(input._keras_shape[DIM2_AXIS] \
        / residual._keras_shape[DIM2_AXIS])
    stride_dim3 = ceil(input._keras_shape[DIM3_AXIS] \
        / residual._keras_shape[DIM3_AXIS])
    equal_channels = residual._keras_shape[CHANNEL_AXIS] \
        == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
            or not equal_channels:
        shortcut = Conv3D(
            filters=residual._keras_shape[CHANNEL_AXIS],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_initializer="he_normal", padding="valid",
            kernel_regularizer=l2(1e-4)
            )(input)
    return add([shortcut, residual])
