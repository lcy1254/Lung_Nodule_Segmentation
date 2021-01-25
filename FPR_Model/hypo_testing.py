import pandas as pd
import matplotlib.pyplot as plt
import math

category = 'Diameter Axial Plane'        #CHOOSE CATEGORY (eg. Subtlety, Diameter Axial Plane, etc.)
thresh = 10                              #CHOOSE THRESHOLD

FilePath = '/Users/ChaeyoungLee/Downloads/inference/iter_22501/specialeval/Attributes copy.xlsx'   #Enter file path to spreadsheet
df = pd.read_excel(FilePath)

size = df[['Detection Bool', category]]
#size = size.to_dict(orient = 'list')

detectDF = size.groupby(['Detection Bool']).get_group('Detected')
undetectDF = size.groupby(['Detection Bool']).get_group('Not Detected')

detect = detectDF[[category]]
undetect = undetectDF[[category]]

#change name
smalltrue = 0
smallfalse = 0
largetrue = 0
largefalse = 0

for size in detect[category]:
    if size >= thresh:
        largetrue+=1
    elif size < thresh:
        smalltrue+=1

for size in undetect['Diameter Axial Plane']:
    if size >= thresh:
        largefalse+=1
    elif size < thresh:
        smallfalse+=1

print('smalltrue:' + str(smalltrue))
print('smallfalse:' + str(smallfalse))
print('largetrue:' + str(largetrue))
print('largefalse:' + str(largefalse))

p1 = largetrue/(largetrue+largefalse)
p2 = smalltrue/(smalltrue+smallfalse)
p = (largetrue+smalltrue)/(largetrue+smalltrue+largefalse+smallfalse)
n1 = largetrue+largefalse
n2 = smalltrue+smallfalse

print('p1: ' + str(p1))
print('p2: ' + str(p2))
print('p: ' + str(p))
print('N1: ' + str(n1))
print('N2: ' + str(n2))

z = abs((p1-p2)/math.sqrt(p*(1-p)*(1/n1+1/n2)))
print('z-score: ' + str(z))
thresh = float(input('Find the z-score of your significance level and enter it here: '))
if z<thresh:
    print('Since the z-score is lower than the significance level, fail to reject the null hypothesis.')
elif z>thresh:
    print('Since the z-score is greater than the significance level, can reject the null hypothesis.')
elif z == thresh:
    print('z-score is equal to the significance level.')
