import sys
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import label, generate_binary_structure
import os
import detection_accuracy as da
import re

#Usage: test_detection_acc.py [gtDir] [predDir]
#example: gtDir = '~/lung_nifti_files', predDir = '~/inference'
#from gtDir get files called 'labels-{}.nii.gz'.format(pid)
#from predDir path: 'iter_{}/prediction/inferred_volume_{}_model_iter_{}.nii.gz'.format(iter, pid, iter)

#get gt and pred data location
gtDir = sys.argv[1]
predDir = sys.argv[2]

assert os.path.isdir(gtDir)
assert os.path.isdir(predDir)

'''
#get output location
if len(sys.argv) == 4:
    outDir = os.path.join(sys.argv[3], 'detection_accuracy')
else:
    outDir = os.path.join(predDir, 'detection_accuracy')

if not os.path.isdir(outDir):
    os.mkdir(outDir)

assert os.path.isdir(outDir)

confusion_matrix = np.zeros((2, 2))
dice = float(0)
dicewoFP = float(0)
img_count = 0'''
    
for predIter in os.listdir(predDir):
    if 'iter_' in predIter and not os.path.isfile(os.path.join(predDir, predIter, 'eval', 'confusion.txt')):
        confusion_matrix = np.zeros((2, 2))
        dice = float(0)
        dicewoFP = float(0)
        img_count = 0
        for predFile in os.listdir(os.path.join(predDir, predIter, 'prediction')):
            if '.nii.gz' in predFile:
                img_num = (re.findall(r'[0-9]+', predFile))[0]
                print(img_num)

                predimg = nib.load(os.path.join(predDir, predIter, 'prediction', predFile))
                gtimg = nib.load(os.path.join(gtDir, 'labels-{}.nii.gz'.format(img_num)))
                img_count += 1

                s = np.ones((3,3,3))
                labeled, nod_num = label(gtimg.dataobj, s)
                nods = {}
                        
                for i in range(nod_num):
                    nods[i] = da.Nod(labeled, i + 1)

                confusion_matrix += da.Nod.computeConfusion(nods, gtimg.dataobj, predimg.dataobj)
                dice += da.Nod.DetectionDice(gtimg.dataobj, predimg.dataobj)
                dicewoFP += da.Nod.DetectionDiceWoFP(gtimg.dataobj, predimg.dataobj)
        finalDice = dice/img_count
        finalDicewoFP = dicewoFP/img_count
        print(confusion_matrix)
        print(finalDice)
        print(finalDicewoFP)
        print(img_count)
        
        outDir = os.path.join(predDir, predIter, 'eval')
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
        
        assert os.path.isdir(outDir)

        paths = {}

        for a in ['confusion', 'dice', 'woFP']:
            paths[a] = os.path.join(outDir, '{}.txt'.format(a))
            count = 0
            while os.path.isfile(paths[a]):
                count += 1
                paths[a] = os.path.join(outDir, '{}_{}.txt'.format(a, count))

        confusionFile = open(paths['confusion'], 'w+')
        confusionFile.write(str(confusion_matrix))
        confusionFile.close()

        diceFile = open(paths['dice'], 'w+')
        diceFile.write(str(finalDice))
        diceFile.close()

        woFP = open(paths['woFP'], 'w+')
        woFP.write(str(finalDicewoFP))
        woFP.close()
