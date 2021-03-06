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

#or can just insert path to script 

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
img_count = 0
'''
    
for predIter in os.listdir(predDir):
    if 'iter_' in predIter:
        confusion_matrix = np.zeros((2, 2))
        dice = float(0)
        dicewoFP = float(0)
        diceTP = float(0)
        img_count = 0
        TP_count = 0
        print(predIter)
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
                
                if nod_num == 0:
                    confusion_matrix += da.Nod.computeConfusion(nods, gtimg.dataobj, predimg.dataobj)
                    img_count -= 1
                    continue
                        
                for i in range(nod_num):
                    nods[i] = da.Nod(labeled, i + 1)

                confusion_matrix += da.Nod.computeConfusion(nods, gtimg.dataobj, predimg.dataobj)
                dice += da.Nod.DetectionDice(gtimg.dataobj, predimg.dataobj)
                dicewoFP += da.Nod.DetectionDicewoFP(gtimg.dataobj, predimg.dataobj)
                
                tempGt = np.array(gtimg.dataobj, dtype=np.int32)
                tempPred = np.array(predimg.dataobj, dtype=np.int32)
                
                if np.sum(tempGt & tempPred)>0:
                    diceTP += da.Nod.DetectionDiceTP(gtimg.dataobj, predimg.dataobj)
                else:
                    TP_count += 1
                
        finalDice = dice/img_count
        finalDicewoFP = dicewoFP/img_count
        finalDiceTP = diceTP/(img_count - TP_count)
        print(confusion_matrix)
        print(finalDice)
        print(finalDicewoFP)
        print(finalDiceTP)
        print(img_count)
        
        outDir = os.path.join(predDir, predIter, 'final_eval')
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
        
        assert os.path.isdir(outDir)

        paths = {}

        for a in ['confusion', 'dice', 'woFP', 'TP']:
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
        
        TP = open(paths['TP'], 'w+')
        TP.write(str(finalDiceTP))
        TP.close()
