import os
import cv2
import copy
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import SimpleITK as stk
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.cluster import KMeans
from skimage import measure

import SimpleITK as sitk
import os

import zipfile

# -------------------------------

with zipfile.ZipFile("lung.zip","r") as zip_ref:
    zip_ref.extractall("ctscan/dt/extract/")

dicom_dir = "ctscan/dt/extract/"
output_dir = "ctscan/dt/"

# Load DICOM series
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
reader.SetFileNames(dicom_names)
image = reader.Execute()

# Save as MHD
output_path = os.path.join(output_dir, 'lung.mhd')
sitk.WriteImage(image, output_path)

# -----------------------

PATH = "ctscan/dt/"
FILE = "lung.mhd"

# -----------------------

def load_mhd(file):
    mhdimage = stk.ReadImage(file)
    ct_scan = stk.GetArrayFromImage(mhdimage)
    origin = np.array(list(mhdimage.GetOrigin()))
    space = np.array(list(mhdimage.GetSpacing()))
    return ct_scan, origin, space

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model = tf.keras.models.load_model("/ctscan/dt/UNet_model.h5", custom_objects={'dice_coef':dice_coef, 'dice_coef_loss':dice_coef_loss})

# read ctscan (masking the ct)

ct, origin, space = load_mhd(PATH+FILE)
# print(ct.shape)
num_z, height, width = ct.shape
ct_norm = cv2.normalize(ct, None, 0, 255, cv2.NORM_MINMAX)   # Normalizing the CT scan
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # CLAHE(Contrast Limited Adaptive Histogram Equalization) filter for enhancing the contrast of an image
ct_norm_improved = []
for layer in ct_norm:
    ct_norm_improved.append(clahe.apply(layer.astype(np.uint8)))  # Applying CLAHE filter to the image
centeral_area = ct_norm_improved[len(ct_norm_improved)//2][100:400, 100:400]
kmeans = KMeans(n_clusters=2).fit(np.reshape(centeral_area, [np.prod(centeral_area.shape), 1]))
centroids = sorted(kmeans.cluster_centers_.flatten())
threshold = np.mean(centroids)
# print(threshold)
lung_masks = []
for layer in ct_norm_improved:
    ret, lung_roi = cv2.threshold(layer, threshold, 255, cv2.THRESH_BINARY_INV)
    lung_roi = cv2.erode(lung_roi, kernel=np.ones([4,4]))
    lung_roi = cv2.dilate(lung_roi, kernel=np.ones([13,13]))
    lung_roi = cv2.erode(lung_roi, kernel=np.ones([8,8]))

    labels = measure.label(lung_roi)        # Labelling different regions in the image
    regions = measure.regionprops(labels)   # Extracting the properties of the regions
    good_labels = []
    for prop in regions:        # Filtering the regions that are not too close to the edges
        B = prop.bbox           # Regions that are too close to the edges are outside regions of lungs
        if B[2]-B[0] < 475 and B[3]-B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    lung_roi_mask = np.zeros_like(labels)
    for N in good_labels:
        lung_roi_mask = lung_roi_mask + np.where(labels == N, 1, 0)

    # Steps to get proper segmentation of the lungs without noise and holes
    contours, hirearchy = cv2.findContours(lung_roi_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    external_contours = np.zeros(lung_roi_mask.shape)
    for i in range(len(contours)):
        if hirearchy[0][i][3] == -1:  #External Contours
            area = cv2.contourArea(contours[i])
            if area>518.0:
                cv2.drawContours(external_contours,contours,i,(1,1,1),-1)
    external_contours = cv2.dilate(external_contours, kernel=np.ones([4,4]))

    external_contours = cv2.bitwise_not(external_contours.astype(np.uint8))
    external_contours = cv2.erode(external_contours, kernel=np.ones((7,7)))
    external_contours = cv2.bitwise_not(external_contours)
    external_contours = cv2.dilate(external_contours, kernel=np.ones((12,12)))
    external_contours = cv2.erode(external_contours, kernel=np.ones((12,12)))

    external_contours = external_contours.astype(np.uint8)      # Final segmentated lungs mask
    lung_masks.append(external_contours)

#extract from masked

extracted_lungs = []
for lung, mask in zip(ct_norm_improved,lung_masks):
    extracted_lungs.append(cv2.bitwise_and(lung, lung, mask=mask))

X = np.array(extracted_lungs)
X.shape
X = (X-127.0)/127.0
X = X.astype(np.float32)
X.dtype
X = np.reshape(X, (len(X), 512, 512, 1))
X.shape
predictions = model.predict(X)
predictions.shape
predictions[predictions>=0.5] = 255
predictions[predictions<0.5] = 0

predictions = predictions.astype(np.uint8)
print(predictions)

pred = list(predictions)
pred = [np.squeeze(i) for i in pred]