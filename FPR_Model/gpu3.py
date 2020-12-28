#based on https://github.com/shervinea/enzynet/blob/master/scripts/architecture/enzynet_adapted.py

import numpy as np
import os
import models
import volume
import tensorflow as tf
import re
from tensorflow import keras
import datetime

from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

import metricsHistory as mh

tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
with tf.device('/GPU:0'):

    mode_run = 'train' #train or test

    ##----------------------------- Parameters -----------------------------------##
    n_classes = 2
    sideLength = 50
    batch_size = 32
    max_epochs = 200
    period_checkpoint = 50
    current_file_name = os.path.basename(__file__)[:-3]

    ##------------------------------ Dataset -------------------------------------##
    #Load list of IDs
    trainingdataDir = '/data/lung_seg/FPR/nodule_files/training'
    validationdataDir = '/data/lung_seg/FPR/nodule_files/validation'

    traininglistIDs = [re.findall(r'[0-9]+', file)[0] for file in os.listDir(trainingdataDir) if '.h5' in file]
    validationlistIDs = [re.findall(r'[0-9]+', file)[0] for file in os.listDir(validationdataDir) if '.h5' in file]

    training_generator = volume.DataGenerator(traininglistIDs, trainingdataDir, batch_size, v_size=sideLength, n_channels=1, n_classes=n_classes, shuffle=True)

    validation_generator = volume.DataGenerator(validationlistIDs, validationdataDir, batch_size, v_size=sideLength, n_channels=1, n_classes=n_classes, shuffle=True)

    ##------------------------------ Model ---------------------------------------##
    #Create
    model = models.VGG16(sideLength)
    #model = models.alexNet(sideLength)
    #model = models.resNet(sideLength)

    #Track accuracy and loss in real-time
    #if jupyter notebook:
    log_dir = "/data/lung_seg/FPR/VGG16/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    #python script
    saving_path = log_dir
    history = mh.MetricsHistory(saving_path=os.path.join(saving_path,'metricsHistory', current_file_name+'.csv'))

    #Checkpoints
    checkpoints = ModelCheckpoint(logdir + '/checkpoints/' + current_file_name + '_{epoch:02d}' + '.hd5f', save_weights_only=True, period=period_checkpoint)

    if mode_run == 'train':
        #Compile
        def scheduler(epoch, lr):
            learning_rate = 0.1
            if epoch > 30:
                learning_rate = 0.02
            if epoch > 50:
                learning_rate = 0.01
            if epoch > 100:
                learning_rate = 0.005
            tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
            return learning_rate
            
        lr_callback = keras.callbacks.LearningRateScheduler(scheduler)
        tensorboard = TensorBoard(log_dir = log_dir, histogram_freq = 1)
        
        model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit_generator(generator=training_generator, epochs=max_epochs, verbose=1, validation_data=validation_generator, callbacks=[history, checkpoints, lr_callback, tensorboard])
        
    K.clear_session()
