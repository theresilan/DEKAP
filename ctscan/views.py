from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os

import requests
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
import stat
import shutil

import zipfile
import subprocess

@api_view(['POST'])
def ctscan(request):
    if request.method == 'POST':
        print("msk")
        os.chmod("/home/renataninagan1/DEKAP/ctscan/dt/", stat.S_IWRITE)
        if os.path.exists("/home/renataninagan1/DEKAP/ctscan/dt/extract"):
            shutil.rmtree("/home/renataninagan1/DEKAP/ctscan/dt/extract")
        if os.path.exists("/home/renataninagan1/DEKAP/ctscan/dt/lung.raw"):
            os.remove("/home/renataninagan1/DEKAP/ctscan/dt/lung.raw")
        if os.path.exists("/home/renataninagan1/DEKAP/ctscan/dt/lung.mhd"):
            os.remove("/home/renataninagan1/DEKAP/ctscan/dt/lung.mhd")
        if os.path.exists("/home/renataninagan1/DEKAP/ctscan/dt/lung.zip"):
            os.remove("/home/renataninagan1/DEKAP/ctscan/dt/lung.zip")

        file_obj = request.FILES.get('file')

        if not file_obj:
            return Response({'error': 'No file was submitted'}, status=status.HTTP_400_BAD_REQUEST)

        # Define the directory where you want to save the uploaded file
        upload_dir = 'ctscan/dt/'
        os.makedirs(upload_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Rename the uploaded file to 'lung.zip'
        file_path = os.path.join(upload_dir, 'lung.zip')

        try:
            with open(file_path, 'wb') as f:
                for chunk in file_obj.chunks():
                    f.write(chunk)
            # here goes the long ass code 
            with zipfile.ZipFile("ctscan/dt/lung.zip","r") as zip_ref:
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

            print(tf.__version__)

            model = tf.keras.models.load_model("ctscan/dt/UNet_model.h5", custom_objects={'dice_coef':dice_coef, 'dice_coef_loss':dice_coef_loss})

            # read ctscan (masking the ct)
            print("BONHOT")
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

            bboxes = []
            centroids = []
            diams = []
            for mask in pred:
                mask = cv2.dilate(mask, kernel=np.ones((5,5)))
                labels = measure.label(mask)
                regions = measure.regionprops(labels)
                bb = []
                cc = []
                dd = []
                for prop in regions:
                    B = prop.bbox
                    C = prop.centroid
                    D = prop.equivalent_diameter_area
                    bb.append((( max(0, B[1]-8), max(0, B[0]-8) ),( min(B[3]+8, 512), min(B[2]+8, 512) )))    # ((x1,y1),(x2,y2))
                    cc.append(C)    # (y,x)
                    dd.append(D)
                bboxes.append(bb)
                centroids.append(cc)
                diams.append(dd)

            bs = []
            mimgs = copy.deepcopy(extracted_lungs)
            for i,(img,boxes) in enumerate(zip(mimgs,bboxes)):
                for rect in boxes:
                    img = cv2.rectangle(img, rect[0], rect[1], (255), 2)

            fpr_model = tf.keras.models.load_model("ctscan/dt/FPR_classifier_model.h5")
            print("load cancer model")
            originals = copy.deepcopy(ct_norm_improved)
            final_boxes = []
            for i,(img,bbox) in enumerate(zip(originals, bboxes)):
                img_boxes = []
                for box in bbox:
                    x1 = box[0][0]
                    y1 = box[0][1]
                    x2 = box[1][0]
                    y2 = box[1][1]
                    if abs(x1-x2) <=50 or abs(y1-y2)<=50:
                        x = (x1+x2)//2
                        y = (y1+y2)//2
                        x1 = max(x-25, 0)
                        x2 = min(x+25, 512)
                        y1 = max(y-25, 0)
                        y2 = min(y+25, 512)
                        imgbox = img[y1:y2,x1:x2]
                        img_boxes.append(imgbox)
                    else:
                        imgbox = img[y1:y2,x1:x2]
                        img_boxes.append(imgbox)
                final_boxes.append(img_boxes)

            fpr_preds = []
            for i in final_boxes:
                each_p = []
                for img in i:
                    if img.shape != (50,50):
                        img = np.resize(img, (50,50))
                    img = img/255.
                    img = np.reshape(img, (1,50,50,1))
                    pred = fpr_model.predict(img)
                    pred = int(pred>=0.5)
                    each_p.append(pred)
                fpr_preds.append(each_p)

            for i in range(len(diams)):
                if len(diams[i]):
                    for j in range(len(diams[i])):
                        diams[i][j] = diams[i][j]*space[0]       # diameters in mm

            final_img_bbox = []
            cancer = []
            df = pd.DataFrame(columns = ['Layer', 'Position (x,y)', 'Diameter (mm)', 'BBox [(x1,y1),(x2,y2)]'])
            e_lungs = copy.deepcopy(ct_norm_improved)
            for i,(img,bbox,preds,cents,dms) in enumerate(zip(e_lungs, bboxes, fpr_preds, centroids, diams)):
                token = False
                for box,pred,cent,dm in zip(bbox,preds,cents,dms):
                    if pred:
                        x1 = box[0][0]
                        y1 = box[0][1]
                        x2 = box[1][0]
                        y2 = box[1][1]
                        img = cv2.rectangle(img, (x1,y1), (x2,y2), (255), 2)
                        dct = pd.DataFrame({'Layer':i, 'Position (x,y)':[f"{cent[::-1]}"], 'Diameter (mm)':dm, 'BBox [(x1,y1),(x2,y2)]':[f"{list(box)}"]})
                        df = pd.concat([df,dct], ignore_index = True)
                        token = True
                final_img_bbox.append(img)
                cancer.append(token)
                df = df.reset_index(drop=True)

                df.head()
                true_count = sum(cancer)

                df.head()  
                true_count = sum(cancer)

                print("sebelum vid, cancer :")
                print(true_count)

                folder = FILE.replace(".mhd", "")
                # os.mkdir(f"/content/gdrive/MyDrive/LUNA/convert/vid")
                # df.to_csv(f"/content/gdrive/MyDrive/LUNA/convert/vid/detections.csv", index=False)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vid = cv2.VideoWriter(f"ctscan/dt/vid/test.mp4", fourcc, 5.0, (512,512), False)
                for i in range(len(final_img_bbox)):
                    img = final_img_bbox[i].copy()
                    img = cv2.putText(img, f"Layer: {i}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
                    vid.write(img)
                vid.release()
                print("beres release")
                # Define the FFmpeg command
                command = [
                    'ffmpeg', '-y', 
                    '-i', "ctscan/dt/vid/test.mp4",
                    '-pix_fmt', 'yuv420p',
                    "ctscan/dt/vid/detections.mp4"
                ]

                # Run the FFmpeg command
                subprocess.run(command, check=True)

                file_path = "ctscan/dt/vid/detections.mp4"
                php_url = "https://dekap.sman5bdg.sch.id/ctscan/verdict.php"

                with open(file_path, 'rb') as file:
                    files = {'file': ('detections.mp4', file)}

                    # Send the file using POST request
                    response = requests.post(php_url, files=files)
            
                print("vid selesai dikirim")

                data_payload = {
                    'cancer': cancer
                }

                # Send the POST request
                response2 = requests.post(
                    php_url,
                    json=data_payload,
                    headers={'Content-Type': 'application/json'}
                )
                # print(response2.status_code)  # Should be 200 if successful
                # print(response2.text) 
                os.makedirs("/home/renataninagan1/DEKAP/ctscan/dt/extract", exist_ok=True)
                print("fin")

            return Response({'message': 'yea', 'file_path': file_path}, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({'error': 'Invalid method'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)
