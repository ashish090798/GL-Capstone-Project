from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import pydicom as dcm
from tensorflow import keras
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from bboxes import extract_bboxes,calculate_bboxes,non_max_suppression,iou_bce_loss,mean_iou
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from matplotlib.patches import Rectangle
import pydicom as dicom
from PIL import Image
import os

imgWidth=224
imgHeight=224
imgChannels=3
imgSize=(imgHeight, imgWidth)
batchSize=64
labelDict={0:'normal', 1:'lung opacity'}
org_img_size=1024
custom_objects={'iou_bce_loss':iou_bce_loss, 'mean_iou':mean_iou}


app = Flask(__name__)
run_with_ngrok(app)

app.config['UPLOAD_FOLDER'] = 'static'

model_path = "Copy of final_model_unet_vgg16_13_10_adam_0.005_64.h5"
@app.route("/") 
def hello():
    return render_template('index.html')

@app.route("/sub",methods = ["POST"]) 
def submit():
    if request.method == "POST":
        img = request.form["input_image"]
        img_data = dcm.read_file(img)
        image = img_data.pixel_array
        img_size = (image.shape[0],image.shape[1])
        image = cv2.resize(image,(224, 224))
        if len(image.shape) !=3 or image.shape[2]!=3:
            image = np.stack([image] * 3, axis=-1)
        model=load_model(model_path, custom_objects=custom_objects, compile=False)
        image1 = preprocess_input(image)
        pred_mask=model.predict(image1.reshape(1,224,224,3))
        pred_org=tf.image.resize(pred_mask, (org_img_size, org_img_size))
        pred_nms=non_max_suppression(np.asarray(pred_org).squeeze())
        if len(pred_nms) == 0:
            output_str = "No pneumonia detected."
        else:
            output_str = "Pneumonia detected."
        image11 = np.stack([img_data.pixel_array] * 3, axis=-1)
        for bbox in pred_nms[0]:
            x,y,w,h=bbox
            image11[y:y+h,x-5:x+5,0],image11[y:y+h,x-5:x+5,1],image11[y:y+h,x-5:x+5,2] = 255,0,0
            image11[y:y+h,x+w-5:x+w+5,0],image11[y:y+h,x+w-5:x+w+5,1],image11[y:y+h,x-5+w:x+5+w,2] = 255,0,0
            image11[y-5:y+5,x:x+w,0],image11[y-5:y+5,x:x+w,1],image11[y-5:y+5,x:x+w,2] = 255,0,0
            image11[y-5+h:y+5+h,x:x+w,0],image11[y-5+h:y+5+h,x:x+w,1],image11[y-5+h:y+5+h,x:x+w,2] = 255,0,0
        im = Image.fromarray(image11)
        if os.path.exists("static\image11.jpg"):
            os.remove("static\image11.jpg")
        im.save("static\image11.jpg")
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image11.jpg')
    return render_template('result.html',s = output_str,user_image = full_filename)

if __name__ == "__main__":
    app.run()