import sys
import nibabel as nib 
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import label, generate_binary_structure
import os
import detection_accuracy
import re

#Usage: test_detection_acc.py [gtDir] [predDir] [outDir]
#example: gtDir = '~/lung_nifti_files', predDir = '~/inference', outDir = ''
#from gtDir get files called 'labels-{}.nii.gz'.format(pid)
#from predDir path: 'iter_{}/prediction/inferred_volume_{}_model_iter_{}.nii.gz'.format(iter, pid, iter)

#get gt and pred data location 
gtDir = sys.argv[1]
predDir = sys.argv[2]

assert os.path.isdir(gtDir)
assert os.path.isdir(predDir)

#get output location 
if len(sys.argv) == 4:
    outDir = os.path.join(sys.argv[3], 'detection_accuracy')
else:
    outDir = os.path.join(predDir, 'detection_accuracy')

if not os.path.isdir(outDir):
    os.mkdir(outDir)

assert os.path.isdir(outDir)

confusion_matrix = np.zeros((2, 2))
dice = 0
dicewoFP = 0
img_count = 0
    
for predIter in os.listdir(predDir):

    for predFile in os.listdir(os.path.join(predDir, predIter, 'prediction')):

        img_num = (re.findall(r'[0-9]+', predFile))[0]

        predimg = nib.load(os.path.join(predDir, predIter, 'prediction', predFile))
        gtimg = nib.load(os.path.join(gtDir, 'labels-{}.nii.gz'.format(img_num)))
        img_count += 1

        s = np.ones((3,3,3))
        labeled, nod_num = label(gtimg.dataobj, s)
        nods = {}
            
        for i in range(nod_num):
            nods[i] = Nod(labeled, i + 1)

        confusion_matrix += Nod.computeConfusion(gtimg.dataobj, predimg.dataobj)
        dice += Nod.DetectionDice(gtimg.dataobj, predimg.dataobj)
        dicewoFP += Nod.DetectionDiceWoFP(gtimg.dataobj, predimg.dataobj)

finalDice = dice/img_count
finalDicewoFP = dicewoFP/img_count

confusionPath = os.path.join(outDir, 'confusion.txt')
dicePath = os.path.join(outDir, 'dice.txt')
FPPath = os.path.join(outDir, 'woFP.txt')

for path, name in zip([confusionPath, dicePath, FPPath], ['confusion', 'dice', 'woFP']):
    count = 0
    while os.path.isfile(path):
        count += 1
        path = os.path.join(outDir, '{}_{}.txt'.format(name, count))

confusionFile = open(confusionPath, 'w+')
confusionFile.write(confusion_matrix)
confusionFile.close()

diceFile = open(dicePath, 'w+')
diceFile.write(finalDice)
diceFile.close()

woFP = open(FPPath, 'w+')
woFP.write(finalDicewoFP)
woFP.close()
