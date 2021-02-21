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
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
'''
tf.config.experimental.set_memory_growth(gpus[1], True)
tf.config.experimental.set_memory_growth(gpus[2], True)
tf.config.experimental.set_memory_growth(gpus[3], True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
'''
testDir = '/data/lung_seg/FPR/nodule_files/testing'
testinglistIDs = [int(re.findall(r'[0-9]+', file)[0]) for file in os.listdir(testDir) if '.h5' in file]

print('whole testing list IDs: ' + str(testinglistIDs))

#epochs = [i for i in range(1, 51)]
epochs = [32]
#CHANGE EPOCHS FOR 200 EPOCH RUNS

a = len(testinglistIDs)
print('total of {} testing images'.format(a))
step = math.ceil(a/16)

batch_size = 2
sideLength = 48

test_generator = testDataGenerator(testinglistIDs, testDir, batch_size=batch_size, v_size=sideLength)
model = models.resNet(sideLength)

for epoch in epochs:
    model.load_weights("/media/data_crypt_2/resNet/2021-02-20_13:22:50/checkpoints/res_aug_{}.hd5f".format(str(epoch).zfill(2)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    prediction = model.predict(test_generator, verbose=1)
    
    savePath = '/media/data_crypt_2/resNet/2021-02-20_13:22:50/predictions'
    if not os.path.isdir(savePath): os.mkdir(savePath)
    f = csv.writer(open(os.path.join(savePath, 'predictions_epoch{}.csv'.format(epoch)), 'w+'))
    f.writerow(prediction)

y_truth = []
for i in testinglistIDs:
    path = os.path.join(testDir, '{}.txt'.format(i))
    with open(path, 'r') as f:
        y = int(f.readline().strip())
        y_truth.append(y)
true_save_path = '/media/data_crypt_2/resNet/2021-02-20_13:22:50/y_truth'
if not os.path.isdir(true_save_path): os.mkdir(true_save_path)
f = csv.writer(open(os.path.join(true_save_path, 'y_truth.csv'), 'w+'))
f.writerow(y_truth)
