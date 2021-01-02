import numpy as np
import tensorflow as tf
import models
from volume import testDataGenerator
import re
import os
import math

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
tf.config.experimental.set_memory_growth(gpus[2], True)
'''
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
'''
testDir = '/data/lung_seg/FPR/nodule_files/testing'
testinglistIDs = [int(re.findall(r'[0-9]+', file)[0]) for file in os.listdir(testDir) if '.h5' in file]

#epochs = [i for i in range(1, 51)]

epochs = [16]
a = len(testinglistIDs)
print('total of {} testing images'.format(a))
step = math.ceil(a/16)

batch_size = 64
sideLength = 48

test_generator = testDataGenerator(testinglistIDs, testDir, batch_size=batch_size, v_size=sideLength)
model = models.VGG16(sideLength)

for epoch in epochs:
    model.load_weights("/data/lung_seg/FPR/no_aug/2021-01-01_18:44:52/checkpoints/no_aug_{}.hd5f".format(str(epoch).zfill(2)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    prediction = model.evaluate(test_generator, verbose=1)
    
    savePath = '/data/lung_seg/FPR/no_aug/2021-01-01_18:44:52/evaluation'
    if not os.path.isdir(savePath): os.mkdir(savePath)
    with open(os.path.join(savePath, 'prediction_epoch{}.txt'.format(epoch)), 'w+') as f:
        f.write(str(prediction))
