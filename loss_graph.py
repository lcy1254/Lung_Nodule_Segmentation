import numpy as np
import sys
import matplotlib.pyplot as plt
import os

#Usage: loss_graph.py [full path to training.txt] [where to start graph (iter)]
#example: loss_graph.py /data/lung_seg/logs/training.txt 98
#if graph should start from first iteration, leave blank 

fileDir = sys.argv[1]
outDir = '/home/lcy'
if len(sys.argv) == 3:
	iter = int(sys.argv[2])
else:
	iter = 0
log = open('{}'.format(fileDir))
word = 'Training loss:'
loss = []
for line in log:
	if word in line:
		loss.append(line.split()[-1])

if iter == 0:
	loss = np.array(loss).astype(np.float)
else:
	loss = np.array(loss).astype(np.float)[iter:]

with open(os.path.join(outDir, 'loss.txt'), 'w+') as f:
    for num in loss:
        f.write('%.5f\n' % num)
'''
x = np.arange(loss.size)

plt.plot(x, loss, 'o')
plt.title('Loss')
plt.show(block=True)
'''
