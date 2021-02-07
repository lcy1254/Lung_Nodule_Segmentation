import os
import re
import numpy as np
import csv 

predDir = '/data/lung_seg/inference'

TPR = []
FPR = []

for predIter in os.listdir(predDir):
    if 'iter_' in predIter:
        try:
            print(str(predIter) + 'try1')
            Fconf = open(os.path.join(predDir, predIter, 'eval', 'confusion.txt'), 'r')
            a = Fconf.read()
            b = re.findall(r'[0-9]+', a)
            tempconf = np.array(b, dtype = 'int').reshape((2,2))
            tp=tempconf[0,0]
            fp=tempconf[0,1]
            TPR.append(tp)
            FPR.append(fp)
        except ValueError:
            print('try2')
            Fconf.close()
            Fconf = open(os.path.join(predDir, predIter, 'eval', 'confusion_1.txt'), 'r')
            a = Fconf.readline().strip()
            a2 = Fconf.readline().strip()
            a = a.strip('[] ')
            a2 = a2.strip('[] ')
            a = a.split()
            a2 = a2.split()
            a = [int(float(val)) for val in a]
            a2 = [int(float(val)) for val in a2]
            tempconf = np.zeros((2,2))
            tp=a[0]
            fp=a[1]
            TPR.append(tp)
            FPR.append(fp)
        except:
            print('try3')
            Fconf.close()
            Fconf = open(os.path.join(predDir, predIter, 'eval', 'confusion_1.txt'), 'r')
            a = Fconf.read()
            b = re.findall(r'[0-9]+', a)
            tempconf = np.array(b, dtype = 'int').reshape((2,2))
            tp=tempconf[0,0]
            fp=tempconf[0,1]
            TPR.append(tp)
            FPR.append(fp)
            
        Fconf.close()

print('TPR: ', str(TPR))
print('FPR: ', str(FPR))

with open('/data/lung_seg/TPvsFP.csv', mode='w+') as f:
    f_writer = csv.writer(f)
    f_writer.writerow(TPR)
    f_writer.writerow(FPR)

file = open('/data/lung_seg/TPvsFP.txt', 'w+')
file.write(str(TPR))
file.write(str(FPR))
file.close()
