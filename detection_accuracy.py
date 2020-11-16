'''
compute detection accuracy
true positive if area of overlap/dice coefficient is >= 40 (CHECK AND FIND GOOD VALUE)
false postive if area of overlap/dice coefficient is <40

find bounding box of true nodules
check dice coefficient of prediction vs truth only inside the bounding box --> determines true positive or false negative
all other nodules are false positives
'''

def cluster(a):
    alist = sorted(a)  #LESSON LEARNED : NEVER WORK DIRECTLY WITH THE INPUTS, BECAUSE IF YOU DO IT CHANGES THE ORIGINAL. ALWAYS MAKE A COPY 
    clusters = []
    
    temp = [alist[0]]
    count = 0
    
    for i in range(1, len(alist)):
        if ((alist[i] - alist[i-1]) <= 5) == True:
            temp.append(alist[i])
        else:
            clusters.append(temp)
            temp = [alist[i]]
    clusters.append(temp)

    #format: [[0,0,0],[0,0,0,0,0,0],[0,0,0]]
    return clusters

def checkConsecutive(l):
    blist = l
    blist = (np.unique(np.array(blist))).tolist()
    return blist == list(range(min(blist), max(blist)+1))

def get_nodules(labels, pref):
    #return bounding box endpoints of all nodules in one scan 
    #if pref = 0, returns number of nodules
    #if pref = 1, returns indexes of nodules 

    #find nonzero indices
    inds = np.nonzero(labels)
    
    #if there is no nodule in image 
    if len(inds[0]) == 0:
        print("no nodule detected")
        if pref == 0:
            return 0
        else:
            return [],[],[]
    
    temp_clus = cluster(inds[0])  #the first dimension is always automatically sorted in np.nonzero function 
    
    dim0_clus = temp_clus
    dim1_clus = []
    dim2_clus = []
    
    for n in range(2):
        start = 0
        for i in range(len(temp_clus)): 
            temp = list(inds[n+1][start:(start+len(temp_clus[i]))])
            exec("{}.append(temp)".format('dim'+str(n+1)+'_clus'))
            start = (start + len(temp_clus[i]))
    
    for i in range(len(dim1_clus)):
        if checkConsecutive(dim1_clus[i]) == True:
            continue
        else:
            a = cluster(dim1_clus[i])
            insert = [[[] for i in range(len(a))] for dim in range(3)]
            #insert[1] = a DO NOT USE 
            for p, value in enumerate(dim1_clus[i]):
                for r in range(len(a)):
                    if (min(a[r]) <= value <= max(a[r])) == True:
                        insert[0][r].append(dim0_clus[i][p])
                        insert[1][r].append(dim1_clus[i][p])
                        insert[2][r].append(dim2_clus[i][p])
                    else:
                        continue 
            for d in range(3):
                exec("{}.pop(i)".format('dim'+str(d)+'_clus'))
                for c in range(len(a)):
                    exec("{}.insert(i,insert[d][c])".format('dim'+str(d)+'_clus'))
    
    for i in range(len(dim2_clus)):
        if checkConsecutive(dim2_clus[i]) == True:
            continue
        else:
            a = cluster(dim2_clus[i])
            insert = [[[] for i in range(len(a))] for dim in range(3)]
            #insert[2] = a
            for p, value in enumerate(dim2_clus[i]):
                for r in range(len(a)):
                    if (min(a[r]) <= value <= max(a[r])) == True:
                        insert[0][r].append(dim0_clus[i][p])
                        insert[1][r].append(dim1_clus[i][p])
                        insert[2][r].append(dim2_clus[i][p])
                    else: 
                        continue 
            for d in range(3):
                exec("{}.pop(i)".format('dim'+str(d)+'_clus'))
                for c in range(len(a)):
                    exec("{}.insert(i,insert[d][c])".format('dim'+str(d)+'_clus'))
    
    assert (len(dim0_clus) == len(dim1_clus) == len(dim2_clus)), "number of nodules do not align"
    if pref == 0:
        return len(dim0_clus)
    elif pref == 1:
        return dim0_clus, dim1_clus, dim2_clus 
    
    '''
    #print("dim0: ", dim0_clus, "\n", "dim1: ", dim1_clus, "\n", "dim2: ", dim2_clus)
    print("number of nodules: " + str(len(dim0_clus)) + " " + str(len(dim1_clus)) + " " + str(len(dim2_clus)))
    flat_list0 = []
    for sublist in dim0_clus:
        for item in sublist:
            flat_list0.append(item)
    flat_list1 = []
    for sublist in dim1_clus:
        for item in sublist:
            flat_list1.append(item)
    flat_list2 = []
    for sublist in dim2_clus:
        for item in sublist:
            flat_list2.append(item)
    print(len(flat_list0), len(flat_list1), len(flat_list2))
    ''' 
    
def boundingBox(labels):
    #for all nodules, just need to input label array 
    dim0, dim1, dim2 = get_nodules(labels, 1)
    nodNum = get_nodules(labels, 0)
    
    #if no nodule in image 
    if nodNum == 0:
        print("no nodule detected")
        return []
    
    bbox = np.zeros((nodNum, 2, labels.ndim), dtype = int)
    for nod in range(nodNum):
        for dim in range(labels.ndim):
            dimInds = []
            exec("dimInds.append(dim{}[nod])".format(str(dim)))
            dimInds = np.array(dimInds).astype(int)
            bbox[nod,:,dim] = [np.amin(dimInds), np.amax(dimInds)]
    return bbox 

def trueNod_detection_accuracy(bboxNod, pred, gt):
    #for one nodule bbox (so need to call this function for every nodule)
    bbox = bboxNod.copy()
    
    #if no nodule in scan 
    if len(bbox) == 0:
        return ("Bbox is empty; no nodule")
    
    for i in range(3):
        bbox[0,i] = bbox[0,i]-10
        bbox[1,i] = bbox[1,i]+10
    predbbox = pred[bbox[0,0]:bbox[1,0], bbox[0,1]:bbox[1,1], bbox[0,2]:bbox[1,2]]
    truebbox = gt[bbox[0,0]:bbox[1,0], bbox[0,1]:bbox[1,1], bbox[0,2]:bbox[1,2]]
    predMask = (predbbox == 1)
    trueMask = (truebbox == 1)
    numIntersect = np.sum(trueMask & predMask)
    acc = float(2*numIntersect/(np.sum(trueMask) + np.sum(predMask)))
    return acc 

def compute_confusion(pred, gt):
    bboxs = boundingBox(gt)
    predNodNum = get_nodules(pred, 0)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    confusion = np.zeros((2, 2))
    
    #if there is no nodule in gt 
    if len(bboxs) == 0:
        print("no nodule in image")
        confusion[0,0] = 0
        confusion[0,1] = predNodNum
        confusion[1,1] = 0
        confusion[1,0] = 0
        return confusion 
        
        
    for bbox in range(len(bboxs)):
        if trueNod_detection_accuracy(bboxs[bbox], pred, gt) > 0.35:
            TP += 1
        else:
            FN += 1
    FP = predNodNum - TP
    confusion[0,0] = TP
    confusion[0,1] = FP
    confusion[1,0] = FN
    confusion[1,1] = TN
    return confusion
