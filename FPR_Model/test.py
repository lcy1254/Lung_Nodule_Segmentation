import numpy as np
import tensorflow as tf
import models
from volume import testDataGenerator
import re
import os
import math
import alex_model
from tensorflow.keras.optimizers import SGD
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
tf.config.experimental.set_memory_growth(gpus[2], True)
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)
#tf.config.experimental.set_memory_growth(gpus[2], True)
#tf.config.experimental.set_memory_growth(gpus[3], True)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

    testDir = '/data/resized_FPR/testing/resized_FPR_testing'  #CHANGE THIS
    testinglistIDs = [int(re.findall(r'[0-9]+', file)[0]) for file in os.listdir(testDir) if '.h5' in file]

    epochs = [i for i in range(1, 51)]    #CHANGE THIS

    a = len(testinglistIDs)
    print('total of {} testing images'.format(a))
    step = math.ceil(a/16)

    batch_size = 128    #CHANGE THIS
    sideLength = 96    #CHANGE THIS

    test_generator = testDataGenerator(testinglistIDs, testDir, batch_size=batch_size, v_size=sideLength)
    model = models.alexNet(sideLength)    #CHANGE THIS

    for epoch in epochs:
        model.load_weights("/data/resized_FPR/models/improved_res_alex_aug/2021-02-16_11:57:24/checkpoints/alex_aug_{}.hd5f".format(str(epoch).zfill(2)))   #CHANGE THIS
        opt = SGD()
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy']) #CHANGE THIS (OPTIMIZER)
        prediction = model.evaluate(test_generator, verbose=1)
        
        savePath = '/data/resized_FPR/models/improved_res_alex_aug/2021-02-16_11:57:24/evaluation'    #CHANGE THIS
        if not os.path.isdir(savePath): os.mkdir(savePath)
        with open(os.path.join(savePath, 'eval_epoch{}.txt'.format(epoch)), 'w+') as f:
            f.write(str(prediction))
