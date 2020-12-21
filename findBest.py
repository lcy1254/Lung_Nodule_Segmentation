import os
import re
import numpy as np

predDir = '/data/lung_seg/inference'

confIter = 0
confIter2 = 0
confIter3 = 0
confusion_matrix = np.zeros((2,2)) #compare true postives
confusion_matrix_2 = np.zeros((2,2))
confusion_matrix_3 = np.zeros((2,2))
diceIter = 0
dice = float(0)
woFPIter = 0
dicewoFP = float(0)
TPIter = 0
TP = float(0)

for predIter in os.listdir(predDir):
    if 'iter_' in predIter:
        try:
            Fconf = open(os.path.join(predDir, predIter, 'best', 'confusion.txt'), 'r')
            a = Fconf.read()
            b = re.findall(r'[0-9]+', a)
            tempconf = np.array(b, dtype = 'int').reshape((2,2))
        except ValueError:
            Fconf.close()
            Fconf = open(os.path.join(predDir, predIter, 'best', 'confusion.txt'), 'r')
            a = Fconf.readline().strip()
            a2 = Fconf.readline().strip()
            a = a.strip('[] ')
            a2 = a2.strip('[] ')
            a = a.split()
            a2 = a2.split()
            a = [int(float(val)) for val in a]
            a2 = [int(float(val)) for val in a2]
            tempconf = np.zeros((2,2))
            tempconf[0,0] = a[0]
            tempconf[0,1] = a[1]
            tempconf[1,0] = a2[0]
            tempconf[1,1] = a2[1] 
        except:
            Fconf.close()
            Fconf = open(os.path.join(predDir, predIter, 'best', 'confusion_1.txt'), 'r')
            a = Fconf.read()
            b = re.findall(r'[0-9]+', a)
            tempconf = np.array(b, dtype = 'int').reshape((2,2))
            Fconf.close()
        
        try:
            Fdice = open(os.path.join(predDir, predIter, 'best', 'dice.txt'), 'r')
            tempdice = float(Fdice.read())
            assert isinstance(tempdice, float) is True
        except:
            Fdice.close()
            Fdice = open(os.path.join(predDir, predIter, 'best', 'dice_1.txt'), 'r')
            tempdice = float(Fdice.read())
            assert isinstance(tempdice, float) is True
        
        try:
            FwoFP = open(os.path.join(predDir, predIter, 'best', 'woFP.txt'), 'r')
            tempwoFP = float(FwoFP.read())
        except:
            FwoFP.close()
            FwoFP = open(os.path.join(predDir, predIter, 'best', 'woFP_1.txt'), 'r')
            tempwoFP = float(FwoFP.read())
        
        try:
            FTP = open(os.path.join(predDir, predIter, 'best', 'TP.txt'), 'r')
            tempTP = float(FTP.read())
        except:
            FTP.close()
            FTP = open(os.path.join(predDir, predIter, 'best', 'woFP_1.txt'), 'r')
            tempTP = float(FTP.read())
            
        Fconf.close()
        Fdice.close()
        FwoFP.close()
        FTP.close()
        
        if tempconf[0,0] > confusion_matrix[0,0]:
            confIter3 = confIter2
            confIter2 = confIter
            confIter = re.findall(r'[0-9]+', predIter)
            confusion_matrix_3 = confusion_matrix_2
            confusion_matrix_2 = confusion_matrix
            confusion_matrix = tempconf
        elif tempconf[0,0] > confusion_matrix_2[0,0]:
            confusion_matrix_3 = confusion_matrix_2
            confusion_matrix_2 = tempconf
            confIter3 = confIter2
            confIter2 = re.findall(r'[0-9]+', predIter)
        elif tempconf[0,0] > confusion_matrix_3[0,0]:
            confusion_matrix_3 = tempconf
            confIter3 = re.findall(r'[0-9]+', predIter)
        if tempdice > dice:
            dice = tempdice
            diceIter = re.findall(r'[0-9]+', predIter)
        if tempwoFP > dicewoFP:
            dicewoFP = tempwoFP
            woFPIter = re.findall(r'[0-9]+', predIter)
        if tempTP > TP:
            TP = tempTP
            TPIter = re.findall(r'[0-9]+', predIter)

print('confusion matrix: \n' + str(confusion_matrix))
print('confusion iter: ' + str(confIter))
print('confusion matrix 2: \n' + str(confusion_matrix_2))
print('confusion iter 2: ' + str(confIter2))
print('confusion matrix 3: \n' + str(confusion_matrix_3))
print('confusion iter 3: ' + str(confIter3))
print('dice: ' + str(dice))
print('dice iter: ' + str(diceIter))
print('woFP: ' + str(dicewoFP))
print('woFP iter: ' + str(woFPIter))
print('TP: ' + str(TP))
print('TP iter: ' + str(TPIter))
