import numpy as np
import tensorflow as tf
import models   #change this
from volume import testDataGenerator
import re
import os
import math
import csv


'''
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
'''

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)
tf.config.experimental.set_memory_growth(gpus[2], True)
tf.config.experimental.set_memory_growth(gpus[3], True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    testDir = '/media/data_crypt_2/temp'
    testinglistIDs = [int(re.findall(r'[0-9]+', file)[0]) for file in os.listdir(testDir) if '.h5' in file]

    print('whole testing list IDs: ' + str(testinglistIDs))

    #epochs = [i for i in range(1, 51)]
    epochs = [34]
    #CHANGE EPOCHS FOR 200 EPOCH RUNS

    a = len(testinglistIDs)
    print('total of {} testing images'.format(a))
    step = math.ceil(a/16)

    batch_size = 64
    sideLength = 48

    test_generator = testDataGenerator(testinglistIDs, testDir, batch_size=batch_size, v_size=sideLength)
    model = models.alexNet(sideLength)

    for epoch in epochs:
        model.load_weights("/data/lung_seg/FPR/alexNet/aug/2021-01-02_01:55:28/checkpoints/alex_aug_{}.hd5f".format(str(epoch).zfill(2)))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        prediction = model.evaluate(test_generator, verbose=1)
        
        savePath = '/media/data_crypt_2/alexNet/eval-final'
        if not os.path.isdir(savePath): os.mkdir(savePath)
        f = csv.writer(open(os.path.join(savePath, 'evaluation_epoch{}.csv'.format(epoch)), 'w+'))
        f.writerow(prediction)
