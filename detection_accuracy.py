#compute confusion matrix & calculate nodule-wise detection accuracies 
class Nod:
    'class for individual nodules'
    s = np.ones((3,3,3))
    
    def __init__(self, orig_labels, class_num):
        labels = np.array(orig_labels)
        self.array = (labels == class_num)
        self.indices = np.nonzero(self.array)
    
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
        vals = np.unique(a)
        predcount = 0
        for val in vals:
            if a.count(val) > 0.1*np.sum(nodMask):
                valMask = (labeledPred == val)
                predcount += np.sum(valMask)
        dice = float(2*numIntersect/(np.sum(nodMask) + predcount))
        return dice
    
    @classmethod 
    def DetectionDice(cls, orig_gt, orig_pred):
        gt = np.array(orig_gt)
        pred = np.array(orig_pred)
        predMask = (pred == 1)
        gtMask = (gt == 1)
        #if no nodule in gt scan
        if np.sum(gtMask) == 0:
            print("no nodule in gt")
            return
        numIntersect = np.sum(gtMask & predMask)
        dice = float(2*numIntersect/(np.sum(gtMask)+np.sum(predMask)))
        return dice 
    
    @classmethod 
    def DetectionDiceWoFP(cls, orig_gt, orig_pred):
        gt = np.array(orig_gt)
        pred = np.array(orig_pred)
        predMask = (pred == 1)
        gtMask = (gt == 1)
        #if no nodule in gt scan
        if np.sum(gtMask) == 0:
            print("no nodule in gt")
            return 
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
    def computeConfusion(cls, orig_gt, orig_pred):
        nod_count = Nod.count(orig_gt)
        pred_nod_count = Nod.pred_count(orig_pred)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        confusion = np.zeros((2, 2))
        for i in range(nod_count):
            acc = nods[i].NodDetectionDice(orig_pred)
            if acc > 0.35:
                TP += 1
            else:
                FN += 1
        FP = pred_nod_count - TP
        assert TP + FP == pred_nod_count
        assert TP + FN == nod_count 
        confusion[0,0] = TP
        confusion[0,1] = FP
        confusion[1,0] = FN
        confusion[1,1] = TN
        return confusion
