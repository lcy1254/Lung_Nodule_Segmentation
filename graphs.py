import pandas as pd
import matplotlib.pyplot as plt
'''
-----------------------------------------------
compute subtlety histogram'''

FilePath = '/Users/ChaeyoungLee/Downloads/inference/iter_11551/eval/Attributes.xlsx'
df = pd.read_excel(FilePath)

size = df[['Detection Bool', 'Subtlety']]
#size = size.to_dict(orient = 'list')

detectDF = size.groupby(['Detection Bool']).get_group('Detected')
undetectDF = size.groupby(['Detection Bool']).get_group('Not Detected')

detect = detectDF[['Subtlety']]
undetect = undetectDF[['Subtlety']]

fig, axs = plt.subplots(1,2,sharey=True,tight_layout=True)
counts, bins, patches = axs[0].hist(detect['Subtlety'], bins=[0,1,2,3,4,5])
axs[0].title.set_text('Detected Nodules')
axs[0].set_xlabel('1(extremely subtle), 5(obvious)')
axs[0].set_xticks(bins)
counts, bins, patches = axs[1].hist(undetect['Subtlety'], bins=[0,1,2,3,4,5])
axs[1].title.set_text('Undetected Nodules')
axs[1].set_xlabel('1(extremely subtle), 5(obvious)')
axs[1].set_xticks(bins)
axs[0].set_ylabel('Nodule Count')

'''
-----------------------------------------------
compute diameter histogram'''

FilePath = '/Users/ChaeyoungLee/Downloads/inference/iter_11551/eval/Attributes.xlsx'
df = pd.read_excel(FilePath)

size = df[['Detection Bool', 'Diameter Axial Plane']]
#size = size.to_dict(orient = 'list')

detectDF = size.groupby(['Detection Bool']).get_group('Detected')
undetectDF = size.groupby(['Detection Bool']).get_group('Not Detected')

detect = detectDF[['Diameter Axial Plane']]
undetect = undetectDF[['Diameter Axial Plane']]

fig, axs = plt.subplots(1,2,sharey=True,tight_layout=True)
axs[0].hist(detect['Diameter Axial Plane'], bins=20, range=[0,40])
axs[0].title.set_text('Detected Nodules')
axs[0].set_xlabel('Diameter (mm)')
axs[1].hist(undetect['Diameter Axial Plane'], bins=20, range=[0,40])
axs[1].title.set_text('Undetected Nodules')
axs[1].set_xlabel('Diameter (mm)')
axs[0].set_ylabel('Nodule Count')

'''
-----------------------------------------------
compute diameter scatter plot'''

FilePath = '/Users/ChaeyoungLee/Downloads/inference/iter_11551/eval/Attributes.xlsx'
df = pd.read_excel(FilePath)

size = df[['Detection Bool', 'Diameter Axial Plane', 'Absolute Max Diameter']]
size = size.to_dict(orient = 'list')

#detectDF = size.groupby(['Detection Bool']).get_group('Detected')
#undetectDF = size.groupby(['Detection Bool']).get_group('Not Detected')

size['detectBool'] = size.pop('Detection Bool')
size['dia'] = size.pop('Diameter Axial Plane')
size['dia2'] = size.pop('Absolute Max Diameter')

size['detectBool'] = [1 if size['detectBool'][i] == 'Detected' else 0 for i in range(len(size['detectBool']))]

plt.scatter(size['dia'], size['dia2'], c=size['detectBool'])
plt.xlim(right=40)
plt.xlim(left=0)
plt.ylim(top=40)
plt.ylim(bottom=0)
plt.xlabel('Diameter (mm)')
plt.ylabel('Absolute Max Diameter (mm)')

'''
-----------------------------------------------
compute subtlety vs diameter scatter plot'''

FilePath = '/Users/ChaeyoungLee/Downloads/inference/iter_11551/eval/Attributes.xlsx'
df = pd.read_excel(FilePath)

size = df[['Detection Bool', 'Diameter Axial Plane', 'Subtlety']]
size = size.to_dict(orient = 'list')

#detectDF = size.groupby(['Detection Bool']).get_group('Detected')
#undetectDF = size.groupby(['Detection Bool']).get_group('Not Detected')

size['detectBool'] = size.pop('Detection Bool')
size['dia'] = size.pop('Diameter Axial Plane')
size['subtlety'] = size.pop('Subtlety')

size['detectBool'] = [1 if size['detectBool'][i] == 'Detected' else 0 for i in range(len(size['detectBool']))]

plt.scatter(size['dia'], size['subtlety'], c=size['detectBool'])
plt.xlabel('Diameter (mm)')
plt.ylabel('Subtlety (1-5 where 5 is obvious)')
