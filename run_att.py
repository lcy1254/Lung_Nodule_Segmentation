import CTLungPreprocessing 
import att
import numpy as np
import sys
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure
import os
import detection_accuracy as da
import re
import pandas as pd
import nodSize as ns

#Usage: run_att.py

gtDir = 'data/lung_seg/lung_nifti_files'
predDir = 'data/lung_seg/blaine/frozen/inference'
outDir = 'data/lung_seg/temp'

assert os.path.isdir(gtDir)
assert os.path.isdir(predDir)

iterList = ['iter_22501', 'iter_21051']

for predIter in os.listdir(predDir):
    if 'iter_' in predIter and predIter in iterList:
        print(predIter) #temporary
        PID = []
        detectAcc = []
        detectBool = []
        vol = []
        nodDiaRoot = []
        nodDiaAxial = []
        nodDiaAbs = []
        subtlety = []
        internalStructure = []
        calcification = []
        sphericity = []
        margin = []
        lobulation = []
        spiculation = []
        texture = []
        malignancy = []
        
        for predFile in os.listdir(os.path.join(predDir, predIter, 'prediction')):
            if '.nii.gz' in predFile:
                img_num = (re.findall(r'[0-9]+', predFile))[0]
                
                predimg = nib.load(os.path.join(predDir, predIter, 'prediction', predFile))
                gtimg = nib.load(os.path.join(gtDir, 'labels-{}.nii.gz'.format(img_num)))
                
                s = np.ones((3,3,3))
                labeled, nod_num = label(gtimg.dataobj, s)
                nods = {}
                
                for i in range(nod_num):
                    nods[i] = da.Nod(labeled, i + 1)
                
                att.getAtts(nods, img_num)
                for i in range(nod_num):
                    #patient ID
                    PID.append(img_num)
                    
                    #detection accuracies & bool
                    temp = nods[i].NodDetectionDice(predimg.dataobj)
                    detectAcc.append(temp)
                    if temp > 0.35:
                        detectBool.append('Detected')
                    else:
                        detectBool.append('Not Detected')
                    
                    #nodule size & volume
                    tempvol = ns.nodVol(nods[i].array)
                    vol.append(tempvol)
                    tempDiaRoot = ns.nodSizeRoot(nods[i].array)
                    nodDiaRoot.append(tempDiaRoot)
                    tempDiaAxial = ns.nodDia(nods[i].array)
                    nodDiaAxial.append(tempDiaAxial)
                    tempDiaAbs = ns.nodAbsDia(nods[i].array)
                    nodDiaAbs.append(tempDiaAbs)
                    
                    #nodule attributes
                    subtlety.append(nods[i].subtlety)
                    internalStructure.append(nods[i].internalStructure)
                    calcification.append(nods[i].calcification)
                    sphericity.append(nods[i].sphericity)
                    margin.append(nods[i].margin)
                    lobulation.append(nods[i].lobulation)
                    spiculation.append(nods[i].spiculation)
                    texture.append(nods[i].texture)
                    malignancy.append(nods[i].malignancy)
        
        outDir = os.path.join(outDir, predIter)
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
        
        dict = {'PatientID': PID, 'Detection Accuracy': detectAcc, 'Detection Bool': detectBool, 'Volume': vol, 'Diameter (comp.w/sqrt)': nodDiaRoot, 'Diameter Axial Plane': nodDiaAxial, 'Absolute Max Diameter': nodDiaAbs,'Subtlety': subtlety, 'Internal Structure': internalStructure, 'Calcification': calcification, 'Sphericity': sphericity, 'Margin': margin, 'Lobulation': lobulation, 'Spiculation': spiculation, 'Texture': texture, 'Malignancy': malignancy}
        
        df = pd.DataFrame(dict)
        finalOutDir = os.path.join(outDir, 'Attributes.xlsx')
        df.to_excel(finalOutDir, float_format="%.2f")
