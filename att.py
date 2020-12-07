import CTLungPreprocessing as ct
import pylidc as pl
import numpy as np
import pandas as pd

#nods --> dictionary of objects

def getAtts(nods, pid):
    assert type(nods) is dict
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == 'LIDC-IDRI-{}'.format(str(pid).zfill(4))).first()
    image = preprocess(scan)
    lung_mask = segment_lung_mask(image, True)
    x1, x2, y1, y2, z1, z2 = create_bounding_box(lung_mask)
    shape = scan.to_volume().shape
    
    allNodules = scan.cluster_annotations()
    nodules = [allNodules[i][0] for i in range(len(allNodules))]   #weakness -- only using one annotation when four is provided
    assert len(nodules) == len(allNodules)
    assert len(nodules) == len(nods)

    for a in range(len(nodules)):
        nod_mask, nod_bbox, ann_masks = consensus(allNodules[a])
        scan_vol = np.zeros(shape)
        scan_vol[nod_bbox] = nod_mask
        temp_vol = reshape_mask(scan_vol, image)
        final_vol = temp_vol[x1:x2, y1:y2, z1:z2]
        for b in range(len(nods)):
            numIntersect = np.sum(final_vol & nods[b].array)
            if numIntersect == np.sum(final_vol):
                nods[b].subtlety = int(nodules[a].subtlety)
                nods[b].internalStructure = int(nodules[a].internalStructure)
                nods[b].calcification = int(nodules[a].calcification)
                nods[b].sphericity = int(nodules[a].sphericity)
                nods[b].margin = int(nodules[a].margin)
                nods[b].lobulation = int(nodules[a].lobulation)
                nods[b].spiculation = int(nodules[a].spiculation)
                nods[b].texture = int(nodules[a].texture)
                nods[b].malignancy = int(nodules[a].malignancy)
