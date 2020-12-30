import os

dir = {'training': '/data/lung_seg/FPR/nodule_files/training', 'validation': '/data/lung_seg/FPR/nodule_files/validation', 'testing': '/data/lung_seg/FPR/nodule_files/testing'}

training1count = 0
training0count = 0
val1count = 0
val0count = 0
test1count = 0
test0count = 0

def read_txt_file(filepath):
    '''read and load class'''
    with open(filepath, 'r') as f:
        label = int(f.readline().strip())
    return label

for key in dir:
    for file in os.listdir(dir[key]):
        if '.txt' in file:
            label = read_txt_file(os.path.join(dir[key], file))
            if key == 'training':
                if label == 0:
                    training0count+=1
                elif label == 1:
                    training1count+=1
            elif key == 'validation':
                if label == 0:
                    val0count+=1
                elif label == 1:
                    val1count+=1
            elif key == 'testing':
                if label == 0:
                    test0count+=1
                elif label == 1:
                    test1count+=1

print("training has {} 0's false positives; {} 1's true positives".format(training0count, training1count))
print("validation has {} 0's false positives; {} 1's true positives".format(val0count, val1count))
print("testing has {} 0's false positives; {} 1's true positives".format(test0count, test1count))
