import os
import re
import numpy as np

predDir = '/data/lung_seg/inference'

iternames = os.listdir(predDir)
nums = [re.findall(r'[0-9]+', name)[0] for name in iternames if 'iter_' in name]
nums.sort()
iternames = ['iter_{}'.format(num) for num in nums]
dice = []
woFP = []
TP = []
iter = []
confusion_matrix = np.zeros((2,2))
for itername in iternames:
    try:
        Fconf = open(os.path.join(predDir, itername, 'best', 'confusion.txt'), 'r')
        a = Fconf.read()
        b = re.findall(r'[0-9]+', a)
        tempconf = np.array(b, dtype = 'int').reshape((2,2))
    except ValueError:
        Fconf.close()
        Fconf = open(os.path.join(predDir, itername, 'best', 'confusion.txt'), 'r')
        a = Fconf.readline().strip()
        a2 = Fconf.readline().strip()
        a = a.strip('[] ')
        a2 = a2.strip('[] ')
        a = a.split()
        a2 = a2.split()
        a = [int(float(val)) for val in a]
        a2 = [int(float(val)) for val in a2]
        tempconf = np.zeros((2,2))
        tempconf[0,0] = int(a[0])
        tempconf[0,1] = int(a[1])
        tempconf[1,0] = int(a2[0])
        tempconf[1,1] = int(a2[1])

    try:
        Fdice = open(os.path.join(predDir, itername, 'best', 'dice.txt'), 'r')
        tempdice = float(Fdice.read())
        assert isinstance(tempdice, float) is True
    except:
        Fdice.close()
        Fdice = open(os.path.join(predDir, itername, 'best', 'dice_1.txt'), 'r')
        tempdice = float(Fdice.read())
        assert isinstance(tempdice, float) is True

    try:
        FwoFP = open(os.path.join(predDir, itername, 'best', 'woFP.txt'), 'r')
        tempwoFP = float(FwoFP.read())
    except:
        FwoFP.close()
        FwoFP = open(os.path.join(predDir, itername, 'best', 'woFP_1.txt'), 'r')
        tempwoFP = float(FwoFP.read())

    try:
        FTP = open(os.path.join(predDir, itername, 'best', 'TP.txt'), 'r')
        tempTP = float(FTP.read())
    except:
        FTP.close()
        FTP = open(os.path.join(predDir, itername, 'best', 'woFP_1.txt'), 'r')
        tempTP = float(FTP.read())
        
    Fconf.close()
    Fdice.close()
    FwoFP.close()
    FTP.close()
    
    iter.append(re.findall(r'[0-9]+', itername)[0])
    dice.append(tempdice)
    woFP.append(tempwoFP)
    TP.append(tempTP)
    confusion_matrix += tempconf

with open('/home/lcy/dice.txt', 'w+') as f:
    for num in dice:
        f.write('%s\n' % num)
with open('/home/lcy/iter.txt', 'w+') as f:
    for num in iter:
        f.write('%s\n' % num)
with open('/home/lcy/woFP.txt', 'w+') as f:
    for num in woFP:
        f.write('%s\n' % num)
with open('/home/lcy/TP.txt', 'w+') as f:
    for num in TP:
        f.write('%s\n' % num)
with open('/home/lcy/confusion.txt', 'w+') as f:
    f.write(str(confusion_matrix))
