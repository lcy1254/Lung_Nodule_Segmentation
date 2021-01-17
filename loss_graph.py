import numpy as np
import sys
import matplotlib.pyplot as plt

#Usage: loss_graph.py [full path to training.txt] [where to start graph (iter)]
#example: loss_graph.py /data/lung_seg/logs/training.txt 98
#if graph should start from first iteration, leave blank 

fileDir = sys.argv[1]
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

x = np.arange(loss.size)

plt.plot(x, loss, 'o')
plt.title('Loss')
plt.show(block=True)

