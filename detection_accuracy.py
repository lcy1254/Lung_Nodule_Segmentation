import numpy as np
from scipy.ndimage import label, generate_binary_structure
import nodSize as size

#compute confusion matrix & calculate nodule-wise detection accuracies
class Nod:
    'class for individual nodules'
    s = np.ones((3,3,3))
    
    def __init__(self, orig_labels, class_num):
        labels = np.array(orig_labels)
        self.array = (labels == class_num)
        self.indices = np.nonzero(self.array)
        self.subtlety = 0
        self.internalStructure = 0
        self.calcification = 0
        self.sphericity = 0
        self.margin = 0
        self.lobulation = 0
        self.spiculation = 0
        self.texture = 0
        self.malignancy = 0
    
    @classmethod
    def count(cls, orig_gt):
        gt = np.array(orig_gt)
        arr, num = label(orig_gt, Nod.s)
        return num
    
    @classmethod
    def pred_count(cls, orig_pred):
        pred = np.array(orig_pred)
        arr, num = label(pred, Nod.s)
        return num
    
    def NodDetectionDice(self, orig_pred):
        pred = np.array(orig_pred)
        predMask = (pred == 1)
        nodMask = self.array
        numIntersect = np.sum(nodMask & predMask)
        #in case predicted nodule may be bigger than true nodule --> prevent false 100% accuracies
        #basically this is replacing the role of the bounding box that I used previously
        labeledPred, prednods = label(predMask, Nod.s)
        x,y,z = np.nonzero(nodMask & predMask)
        #in case two nodules are predicted in the area of one true nodule
        #should delete? because this would significantly lower the accuracy
        a = [labeledPred[x[i],y[i],z[i]] for i in range(len(x))]
        vals = list(np.unique(a))
        for val in vals:
            if a.count(val)< 0.2*numIntersect:
                vals.remove(val)
            else:
                continue
        predcount = 0
        for val in vals:
            if a.count(val) > 0.1*np.sum(nodMask):
                valMask = (labeledPred == val)
                predcount += np.sum(valMask)
        dice = float(2*numIntersect/(np.sum(nodMask) + predcount))
        return dice
    
    @classmethod
    def DetectionDice(cls, orig_gt, orig_pred):
        #exclude cases where # of nod = 0
        gt = np.array(orig_gt)
        pred = np.array(orig_pred)
        predMask = (pred == 1)
        gtMask = (gt == 1)
        #if no nodule in gt scan
        if np.sum(gtMask) == 0:
            print("no nodule in gt")
            return float(0) #ask what I should return
            #or should I just skip these and not include them
        numIntersect = np.sum(gtMask & predMask)
        dice = float(2*numIntersect/(np.sum(gtMask)+np.sum(predMask)))
        return dice
    
    @classmethod
    def DetectionDiceTP(cls, orig_gt, orig_pred, best=None):
        #exclude cases where # of nod = 0
        if best == False or best == None:
            gt = np.array(orig_gt)
            gtMask = (gt == 1)
        elif best == True:
            gt = np.array(orig_gt)
            gtMask = (Nod.bestMask(gt) > 0)
        pred = np.array(orig_pred)
        predMask = (pred == 1)
        #if no nodule in gt scan
        if np.sum(gtMask) == 0:
            print("no nodule in gt")
            return float(1) #ask what I should return
            #or should I just skip these and not include them
        #get num of pixels of intersection -- true positive
        numIntersect = np.sum(gtMask & predMask)
        #only get num of pixels in nodules that are TP -- exclude FP
        labeledPred, prednods = label(predMask, Nod.s)
        x,y,z = np.nonzero(gtMask & predMask)
        #a drawback is that a FP nodule might be included with just a pixel's overlap
        a = [labeledPred[x[i],y[i],z[i]] for i in range(len(x))]
        vals = np.unique(a)
        predcount = 0
        for val in vals:
            valMask = (labeledPred == val)
            predcount += np.sum(valMask)
        labeledGt, gtnods = label(gtMask, Nod.s)
        x1,y1,z1 = np.nonzero(gtMask & predMask)
        a1 = [labeledGt[x1[i], y1[i], z1[i]] for i in range(len(x1))]
        vals1 = np.unique(a1)
        gtcount = 0
        for val in vals1:
            valMask = (labeledGt == val)
            gtcount += np.sum(valMask)
        diceTP = float(2*numIntersect/(gtcount + predcount))
        return diceTP
    
    @classmethod
    def DetectionDicewoFP(cls, orig_gt, orig_pred, best=None):
        #exclude cases where # of nod = 0
        if best == False or best == None:
            gt = np.array(orig_gt)
            gtMask = (gt == 1)
        elif best == True:
            gt = np.array(orig_gt)
            gtMask = (Nod.bestMask(gt) > 0)
        pred = np.array(orig_pred)
        predMask = (pred == 1)
        #if no nodule in gt scan
        if np.sum(gtMask) == 0:
            print("no nodule in gt")
            return float(1) #ask what I should return
            #or should I just skip these and not include them
        #get num of pixels of intersection -- true positive
        numIntersect = np.sum(gtMask & predMask)
        #only get num of pixels in nodules that are TP -- exclude FP
        labeledPred, prednods = label(predMask, Nod.s)
        x,y,z = np.nonzero(gtMask & predMask)
        #a drawback is that a FP nodule might be included with just a pixel's overlap
        a = [labeledPred[x[i],y[i],z[i]] for i in range(len(x))]
        vals = np.unique(a)
        predcount = 0
        for val in vals:
            valMask = (labeledPred == val)
            predcount += np.sum(valMask)
        diceWoFP = float(2*numIntersect/(np.sum(gtMask) + predcount))
        return diceWoFP
    
    @classmethod
    def computeConfusion(cls, nods, orig_gt, orig_pred, best=None):
        nod_count = 0
        pred_nod_count = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        confusion = np.zeros((2, 2))
        if best == False or best == None:
            nod_count = Nod.count(orig_gt)
            pred_nod_count = Nod.pred_count(orig_pred)
            for i in range(nod_count):
                acc = nods[i].NodDetectionDice(orig_pred)
                if acc > 0.15:
                    TP += 1
                else:
                    FN += 1
            FP = pred_nod_count - TP
            assert TP + FP == pred_nod_count
            assert TP + FN == nod_count
        elif best == True:
        #gt nodules bigger than or equal to 10mm in diameter (however, includes all predicted nodules regardless of size)
            for i in nods:
                if size.nodDia(nods[i].array) >= 10:
                    nod_count += 1
                    acc = nods[i].NodDetectionDice(orig_pred)
                    if acc > 0.15:
                        TP += 1
                    else:
                        FN += 1
                else:
                    continue
            pred_nod_count = Nod.pred_count(orig_pred)
            FP = pred_nod_count - TP
        confusion[0,0] = TP
        confusion[0,1] = FP
        confusion[1,0] = FN
        confusion[1,1] = TN
        return confusion
        
    @classmethod
    def best_nod_count(cls, orig_array):
        arr = np.array(orig_array)
        arr = (arr > 0)
        labeledArr, nodCount = label(arr, Nod.s)
        count = 0
        if nodCount == 0:
            return 0
        for i in range(nodCount):
            nodArr = (labeledArr == (i+1))
            if size.nodDia(nodArr) >= 10:
                count += 1
        return count
    
    @classmethod
    def bestMask(cls, orig_array):
        arr = np.array(orig_array)
        arr = (arr > 0)
        labeledArr, nodCount = label(arr, Nod.s)
        if nodCount == 0:
            return arr
        for i in range(nodCount):
            temp = (labeledArr == (i+1))
            if size.nodDia(temp) < 10:
                a = np.argwhere(temp==1)
                for i in range(len(a)):
                    arr[tuple(a[i])] = 0
        return arr
