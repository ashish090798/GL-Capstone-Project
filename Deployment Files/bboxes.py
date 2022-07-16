import numpy as np
import pandas as pd

# import wandb
# from wandb.keras import WandbCallback

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from matplotlib.patches import Rectangle
import pydicom as dicom
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.measure import label, regionprops
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import precision_score, recall_score
from tensorflow import keras
import tensorflow as tf

import albumentations as A


def extract_bboxes(mask):
    lbl_0 = label(mask)
    bboxes = regionprops(lbl_0)
    img_boxes=[]
    for bbox in bboxes:
        y1, x1, y2, x2 = bbox.bbox
        w, h = x2-x1, y2-y1
        if w > mask.shape[1]*0.1 and h > mask.shape[0]*0.1: # keep bounding boxes that are atleast 20% of the image size elimiating small areas detected
            img_boxes.append([x1,y1,w,h])
    
    return img_boxes

def calculate_bboxes(bboxes, overlapThresh):
    bboxes=np.array(bboxes)
    pick = []
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    w = bboxes[:,2]
    h = bboxes[:,3]
    x2=x1+w
    y2=y1+h
    
    area = w*h
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return list(bboxes[pick].astype("int"))

def non_max_suppression(predmasks, maskTh=0.5, overlapThresh=0.5):
    predABBoxes, predBBoxes=[], []
    bboxes=extract_bboxes(predmasks[:,:] > maskTh)
    if len(bboxes) == 0:
        predABBoxes.append([])
    else:
        predABBoxes.append(calculate_bboxes(bboxes, overlapThresh))

    for i in range(len(predABBoxes)):
        if len(predABBoxes[i])==0:
            predBBoxes.append(predABBoxes[i])
        else:
            predBBoxes.append([list(y) for y in predABBoxes[i]])
    del predABBoxes
    return predBBoxes

def iou_bce_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred = y_pred>0.5
    intersect = tf.reduce_sum(float(y_true) * float(y_pred), axis=[1])
    union = tf.reduce_sum(float(y_true),axis=[1]) + tf.reduce_sum(float(y_pred),axis=[1])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(float(y_true) * float(y_pred))
    score = (intersection + 1.) / (tf.reduce_sum(float(y_true)) + tf.reduce_sum(float(y_pred)) - intersection + 1.)
    return 1 - score