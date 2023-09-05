import torch
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
from ultralytics import YOLO

model = YOLO('best.pt')
# model.train(data='TheSV.yaml',epochs=10)

st.columns(2)

col1, col2,col3 = st.columns(3)

with col1:
    st.header("Import image")
    img_file_buffer = st.file_uploader("Choose a file")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image) # if you want to pass it to OpenCV
        st.image(image, caption="The caption", use_column_width=True)
        
with col3:
    st.header("Retrieve Information:")
    button = st.button("Retrieve Information:")
    if button:        
        # OCR
        st.write('running...')
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained']=False
        config['device'] = 'cpu'
        
        # detect information
        results = model.predict(image)
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            probs = result.probs
        
        name_text = []
        year_text = []
        faculty_text = []
        ID_text = []
        face_img = []
        
        detector = Predictor(config)
        cnt = 0
        for i in range(len(boxes)):
            if boxes.cls[i] == 0:
                cnt += 1
            if boxes.cls[i] == 1:
                name_arr = img_array[int(boxes.xyxy[i][1]):int(boxes.xyxy[i][3]),int(boxes.xyxy[i][0]):int(boxes.xyxy[i][2])]
                name_img = Image.fromarray(name_arr)
                name_text = detector.predict(name_img)
            elif boxes.cls[i] == 2:
                year_arr = img_array[int(boxes.xyxy[i][1]):int(boxes.xyxy[i][3]),int(boxes.xyxy[i][0]):int(boxes.xyxy[i][2])]
                year_img = Image.fromarray(year_arr)
                year_text = detector.predict(year_img)
            elif boxes.cls[i] == 3:
                faculty_arr = img_array[int(boxes.xyxy[i][1]):int(boxes.xyxy[i][3]),int(boxes.xyxy[i][0]):int(boxes.xyxy[i][2])]
                faculty_img = Image.fromarray(faculty_arr)
                faculty_text = detector.predict(faculty_img)
            elif boxes.cls[i] == 4:
                ID_arr = img_array[int(boxes.xyxy[i][1]):int(boxes.xyxy[i][3]),int(boxes.xyxy[i][0]):int(boxes.xyxy[i][2])]
                ID_img = Image.fromarray(ID_arr)
                ID_text = detector.predict(ID_img)

            elif boxes.cls[i] == 5:
                face_arr = img_array[int(boxes.xyxy[i][1]):int(boxes.xyxy[i][3]),int(boxes.xyxy[i][0]):int(boxes.xyxy[i][2])]
                face_img = Image.fromarray(face_arr).resize((400,600))
        if cnt >1: 
            st.write('Alert: Must be just 1 UIT Card at image')
        else:
            if name_text:
                st.write('name: ',name_text)
            else:
                st.write('name: ')
            if year_text:
                st.write('year: ',year_text)
            else: 
                st.write('name: ')
            if faculty_text:
                st.write('faculty: ',faculty_text)
            else:
                st.write('faculty: ')
            if ID_text:
                st.write('ID: ',ID_text)    
            else:
                st.write('ID: ')
            if face_img:
                st.image(face_img,'face')