import numpy as np
import tensorflow as tf
import models
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
    testDir = '/data/lung_seg/FPR/nodule_files/testing'
    testinglistIDs = [int(re.findall(r'[0-9]+', file)[0]) for file in os.listdir(testDir) if '.h5' in file]

    print('whole testing list IDs: ' + str(testinglistIDs))

    epochs = [i for i in range(1, 51)]
    #CHANGE EPOCHS FOR 200 EPOCH RUNS

    a = len(testinglistIDs)
    print('total of {} testing images'.format(a))
    step = math.ceil(a/16)

    batch_size = 256
    sideLength = 48

    test_generator = testDataGenerator(testinglistIDs, testDir, batch_size=batch_size, v_size=sideLength)
    model = models.VGG16(sideLength)

    for epoch in epochs:
        model.load_weights("/media/data_crypt_2/no_aug/2021-01-17_00:18:41/checkpoints/no_aug_{}.hd5f".format(str(epoch).zfill(2)))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        prediction = model.evaluate(test_generator, verbose=1)
        
        savePath = '/media/data_crypt_2/no_aug/2021-01-17_00:18:41/eval'
        if not os.path.isdir(savePath): os.mkdir(savePath)
        f = csv.writer(open(os.path.join(savePath, 'eval_epoch{}.csv'.format(epoch)), 'w+'))
        f.writerow(prediction)
