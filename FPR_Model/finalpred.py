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
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)
tf.config.experimental.set_memory_growth(gpus[2], True)
tf.config.experimental.set_memory_growth(gpus[3], True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
'''

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

testDir = '/media/data_crypt_2/temp'
testinglistIDs = [int(re.findall(r'[0-9]+', file)[0]) for file in os.listdir(testDir) if '.h5' in file]

print('whole testing list IDs: ' + str(testinglistIDs))

#epochs = [i for i in range(1, 51)]
epochs = [33]
#CHANGE EPOCHS FOR 200 EPOCH RUNS

a = len(testinglistIDs)
print('total of {} testing images'.format(a))

batch_size = 2
sideLength = 48

test_generator = testDataGenerator(testinglistIDs, testDir, batch_size=batch_size, v_size=sideLength)
model = models.VGG16(sideLength)

for epoch in epochs:
    model.load_weights("/data/lung_seg/FPR/VGG16/aug2/2021-01-04_03:41:48/checkpoints/vgg_aug_{}.hd5f".format(str(epoch).zfill(2)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    prediction = model.predict(test_generator, verbose=1)
    
    savePath = '/media/data_crypt_2/VGG/VGG_AUG_eval-final'
    if not os.path.isdir(savePath): os.mkdir(savePath)
    f = csv.writer(open(os.path.join(savePath, 'predictions_epoch{}.csv'.format(epoch)), 'w+'))
    f.writerow(prediction)

y_truth = []
for i in testinglistIDs:
    path = os.path.join(testDir, '{}.txt'.format(i))
    with open(path, 'r') as f:
        y = int(f.readline().strip())
        y_truth.append(y)
true_save_path = '/media/data_crypt_2/VGG/y_truth'
if not os.path.isdir(true_save_path): os.mkdir(true_save_path)
f = csv.writer(open(os.path.join(true_save_path, 'y_truth.csv'), 'w+'))
f.writerow(y_truth)
