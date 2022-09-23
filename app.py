import cv2
import streamlit as st
import time
import tensorflow as tf
import numpy as np


st.markdown("<h1 style='text-align: center; color: Aqua;'>$IGN LANGUAGE DETECTION</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: right; color: yellow;'>~sumesh varadharajan</h6>", unsafe_allow_html=True)
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
model2 = tf.keras.models.load_model('DeepVisionModel.h5')
font = cv2.FONT_HERSHEY_SIMPLEX
col1, col2, col3 = st.columns([0.2, 5, 0.2])

# org
org = (320, 480)
fontScale = 1
color = (255, 240, 0)
thickness = 3
if camera.isOpened():
    run = st.checkbox('Click to Run/Stop')
    while run:
        return_value, frame1 = camera.read()
        #FRAME_WINDOW.image(frame1)
        time.sleep(0.0001)

        return_value, frame2 = camera.read()
        #FRAME_WINDOW.image(frame2)

        time.sleep(0.0001)
        image_1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        #image1 = cv2.imdecode(frame1, cv2.IMREAD_COLOR)
        #image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image1 = cv2.resize(image1,(256,256))
        image1 = np.dstack([image1]*3)

        image_2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        #image2 = cv2.imdecode(frame2, cv2.IMREAD_COLOR)
        #image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        image2 = cv2.resize(image2,(256,256))
        image2 = np.dstack([image2]*3)




        absdiff = cv2.absdiff(image1,image2)
        absdiff1 = np.expand_dims(absdiff, axis = 0)
        val = model2.predict(absdiff1)
        absdiff = cv2.resize(absdiff,(750,500))
        image_1 = cv2.resize(image_1,(750,500))

        if val == 0:
             absdiff = cv2.putText(image_1, 'Signed', org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
             FRAME_WINDOW.image(absdiff)
        else:
            
            absdiff = cv2.putText(image_1, 'Unsigned', org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
            FRAME_WINDOW.image(absdiff)
    

    else:
        
        st.markdown("<h4 style='text-align: center; color: gray;'>....</h4>", unsafe_allow_html=True)


else:
    
    st.markdown("<h3 style='text-align: center; color: red;'>Sorry for the trouble we couldn't open the camera</h3>", unsafe_allow_html=True)
    

    
   
