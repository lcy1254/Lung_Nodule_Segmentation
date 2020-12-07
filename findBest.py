import os
import re
import numpy as np

predDir = '/data/lung_seg/inference'

confIter = 0
confusion_matrix = np.zeros((2,2)) #compare true postives
diceIter = 0
dice = float(0)
woFPIter = 0
dicewoFP = float(0)

for predIter in os.listdir(predDir):
    if 'iter_' in predIter:
        Fconf = open(os.path.join(predDir, predIter, 'eval', 'confusion.txt'), 'r')
        Fdice = open(os.path.join(predDir, predIter, 'eval', 'dice.txt'), 'r')
        FwoFP = open(os.path.join(predDir, predIter, 'eval', 'woFP.txt'), 'r')
        
        tempconf = np.array(Fconf.read())
        tempdice = float(Fdice.read())
        tempwoFP = float(FwoFP.read())
        
        if tempconf[0][0] > confusion_matrix[0][0]:
            confusion_matrix = tempconf
            confIter = re.findall(r'[0-9]+', predIter)
        if tempdice > dice:
            dice = tempdice
            diceIter = re.findall(r'[0-9]+', predIter)
        if tempwoFP > dicewoFP:
            dicewoFP = tempwoFP
            woFPIter = re.findall(r'[0-9]+', predIter)

print('confusion matrix:' + confusion_matrix)
print('confusion iter:' + confIter)
print('dice:' + dice)
print('dice iter:' + diceIter)
print('woFP:' + dicewoFP)
print('woFP iter:' + woFPIter)

