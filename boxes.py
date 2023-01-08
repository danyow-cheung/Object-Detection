import numpy as np
import skimage
import matplotlib.pyplot as plt
import os
import layer_utils
import label_utils
import math

from skimage.io import imread
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from layer_utils import anchor_boxes, minmax2centroid, centroid2minmax
from label_utils import index2class, get_box_color

def nms(args,classes,offsets,anchors):
    '''perform NMS(Algorithm 11.12.1)
    Arguments:
        args:               User-defined configurations
        classes(tensor):    Predicted classes
        offsets(tensor):    Predicted offsets
    Returns:
        objects(tensor):    class predictions per anchor
        indexes(tensor):    indexes of detected objects filtered by NMS

        scores(tensor):     array of detected objects scores filtered by NMS
    '''
    # get all non-zero (non-background ) objects 
    objects = np.argmax(classes,axis=1)
    # non-zero indexes are not background 
    nonbg = np.nonzero(object)[0]

    # D and S indexes in Li
    indexes = []
    while True:
        # list of zero probability values
        scores = np.zeros((classes.shape[0],))
        # set probability values of non-background
        scores[nonbg] = np.amax(classes[nonbg],axis=1)

        # max probability given the list 
        score_idx = np.argmax(scores,axis=0)
        score_max = scores[score_idx]

        # get all non max probability & set it as new nonbg 
        nonbg = nonbg[nonbg!= score_idx]

        # if max obj probability is less than threshold ()
        if score_max<args.class_threshold:
            # we are done 
            break
            
        indexes.append(score_idx)
        score_anc = anchors[score_idx]
        score_off = offsets[score_idx][0:4]

        score_box = score_anc+score_off
        score_box = np.expand_dims(score_box,axis=0)
        nonbg_copy = np.copy(nonbg)

        # get all overlapping predictions 
        # perform Non-max suppression
        for idx in nonbg_copy:
            anchor = anchors[idx]
            offset = offsets[idx][0:4]

            box = anchor+offset
            box = np.expand_dims(box,axis=0)
            iou = label_utils.iou(box,score_box)[0][0]
            # if soft NMS is chosen
            if args.soft_nms:
                # adjust score:
                iou = -2 *iou*iou 
                classes[idx] *= math.exp(iou)
            
            # else NMS 
            elif iou>= args.iou_threshold:
                # remove overlapping predictions with iou>threshold
                nonbg = nonbg[nonbg!=idx]
            
        # nothing else to process 
        if nonbg.size ==0:
            break
        # get the array of object scores
        scores = np.zeros((classes.shape[0],))
        scores[indexes] = np.argmax(classes[indexes],axis=1)
        return objects,indexes,scores
        