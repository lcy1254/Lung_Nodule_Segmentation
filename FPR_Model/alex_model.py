def alexNet(sideLength):
    '''AlexNet 3D CNN Implementation'''
    #reduced # of filters 
    input = keras.Input(shape=(sideLength, sideLength, sideLength, 1))
    
    x = layers.Conv3D(filters=96, kernel_size=11, strides=3, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(2, strides=1)(x)
    
    x = layers.Conv3D(128, 5, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(3, strides=2)(x)
    
    x = layers.Conv3D(256, 3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    '''
    x = layers.Conv3D(384, 3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    '''
    x = layers.Conv3D(128, 3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((3,3,3), (2,2,2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(input, output, name='AlexNet')
    return model
'''
version 1 of model

def alexNet(sideLength):
    ''#'AlexNet 3D CNN Implementation'#''
    input = keras.Input(shape=(sideLength, sideLength, sideLength, 1))
    
    x = layers.Conv3D(filters=96, kernel_size=11, strides=3, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(2, strides=1)(x)
    
    x = layers.Conv3D(256, 5, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(3, strides=2)(x)
    
    x = layers.Conv3D(384, 3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    '#''
    x = layers.Conv3D(384, 3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    '#''
    x = layers.Conv3D(256, 3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((3,3,3), (2,2,2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(input, output, name='AlexNet')
    return model
'''
