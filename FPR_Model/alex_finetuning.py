#based on https://github.com/shervinea/enzynet/blob/master/scripts/architecture/enzynet_adapted.py

import numpy as np
import os
import alex_model
import volume
import tensorflow as tf
import re
from tensorflow import keras
import datetime
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K

import metricsHistory as mh

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)
tf.config.experimental.set_memory_growth(gpus[2], True)
tf.config.experimental.set_memory_growth(gpus[3], True)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    #tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    mode_run = 'train' #train or test

    ##----------------------------- Parameters -----------------------------------##
    n_classes = 2
    sideLength = 48
    batch_size = 64
    max_epochs = 50
    period_checkpoint = 1
    class_weight = {0: 0.6, 1: 3.4}
    current_file_name = os.path.basename(__file__)[:-3]

    ##------------------------------ Dataset -------------------------------------##
    #Load list of IDs
    trainingdataDir = '/data/lung_seg/FPR/nodule_files/training'
    validationdataDir = '/data/lung_seg/FPR/nodule_files/validation'

    traininglistIDs = [re.findall(r'[0-9]+', file)[0] for file in os.listdir(trainingdataDir) if '.h5' in file]
    validationlistIDs = [re.findall(r'[0-9]+', file)[0] for file in os.listdir(validationdataDir) if '.h5' in file]

    training_generator = volume.DataGenerator(traininglistIDs, trainingdataDir, v_size=sideLength, batch_size=batch_size, n_channels=1, n_classes=n_classes, shuffle=True)

    validation_generator = volume.DataGenerator(validationlistIDs, validationdataDir, v_size=sideLength, batch_size=batch_size, n_channels=1, n_classes=n_classes, shuffle=True)

    ##------------------------------ Model ---------------------------------------##
    #Create
    model = alex_model.alexNet(sideLength)
    model.summary()

    #Track accuracy and loss in real-time
    #if jupyter notebook:
    log_dir = "/data/lung_seg/FPR/alexNet/dump/" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()

    #python script
    saving_path = log_dir
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)
    saving_path = os.path.join(saving_path, 'metricsHistory')
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)
    history = mh.MetricsHistory(saving_path=os.path.join(saving_path, current_file_name+'.csv'))

    #Checkpoints
    checkpoints = ModelCheckpoint(log_dir + '/checkpoints/' + current_file_name + '_{epoch:02d}' + '.hd5f', save_weights_only=True, period=period_checkpoint)

    if mode_run == 'train':
        #Compile
        '''
        lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, mode='min')
        '''
        def scheduler(epoch, lr):
            learning_rate = 0.001
            if epoch > 5:
                learning_rate = 0.0001
            if epoch > 30:
                learning_rate = 0.00001
            if epoch > 45:
                learning_rate = 0.000005
            tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
            return learning_rate
            
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        tensorboard = TensorBoard(log_dir = log_dir, histogram_freq = 1)
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        model.fit_generator(generator=training_generator, epochs=max_epochs, verbose=1, validation_data=validation_generator, callbacks=[history, checkpoints, lr_callback, tensorboard], class_weight=class_weight)




