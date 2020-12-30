import numpy as np
import tensorflow as tf
import models
from volume import testDataGenerator
import re
import os
import math

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)
tf.config.experimental.set_memory_growth(gpus[2], True)
tf.config.experimental.set_memory_growth(gpus[3], True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    testDir = '/data/lung_seg/FPR/nodule_files/testing'
    testinglistIDs = [int(re.findall(r'[0-9]+', file)[0]) for file in os.listdir(testDir) if '.h5' in file]
    
    epochs = [i for i in range(1, 51)]
    
    a = len(testinglistIDs)
    print('total of {} testing images'.format(a))
    step = math.ceil(a/16)
    
    batch_size = 16
    sideLength = 48
    
    test_generator = testDataGenerator(testinglistIDs, testDir, batch_size=batch_size, v_size=sideLength)
    model = models.alexNet(sideLength)
    
    for epoch in epochs:
        model.load_weights("/data/lung_seg/FPR/alexNet/logs/fit/2020/12/30-00:33:31/checkpoints/gpu0_{}.hd5f".format(str(epoch).zfill(2)))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        prediction = model.evaluate(test_generator, verbose=1)
        
        savePath = '/data/lung_seg/FPR/alexNet/logs/fit/2020/12/30-00:33:31'
        with open(os.path.join(savePath, 'prediction_epoch{}.txt'.format(epoch)), 'w+') as f:
            f.write(str(prediction))
