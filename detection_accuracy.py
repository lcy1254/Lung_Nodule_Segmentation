#compute confusion matrix & calculate nodule-wise detection accuracies 
class Nod:
    'class for individual nodules'
    s=np.array([[[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]],

   [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]],

   [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]]], dtype='uint8')
    
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
    
    def DetectionAccuracy(self, orig_pred):
        pred = np.array(orig_pred)
        predMask = (pred == 1)
        nodMask = self.array
        numIntersect = np.sum(nodMask & predMask)
        #in case predicted nodule may be bigger than true nodule --> prevent false 100% accuracies 
        #basically this is replacing the role of the bounding box that I used previously 
        labeledPred, prednods = label(predMask, Nod.s)
        x,y,z = np.nonzero(nodMask & predMask)
        a = [labeledPred[x[i],y[i],z[i]] for i in range(len(x))]
        vals = np.unique(a)
        predcount = 0
        for val in vals:
            valMask = (labeledPred == val)
            predcount += np.sum(valMask)
        acc = float(2*numIntersect/(np.sum(nodMask) + predcount))
        return acc
    
    @classmethod 
    def computeConfusion(cls, orig_gt, orig_pred):
        nod_count = Nod.count(orig_gt)
        pred_nod_count = Nod.pred_count(orig_pred)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        confusion = np.zeros((2, 2))
        for nod in range(nod_count):
            exec('acc = nod{}.DetectionAccuracy(orig_pred)'.format(str(nod)), locals(), globals())
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
