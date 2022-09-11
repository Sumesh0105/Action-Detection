"""import cv2
import streamlit as st
import time
import tensorflow as tf
import numpy as np


st.title("SIGNED OR UNSIGNED")
st.markdown("<h6 style='text-align: right; color: gray;'>~sumesh varadharajan</h6>", unsafe_allow_html=True)
run = st.checkbox('Click to Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
model2 = tf.keras.models.load_model('DeepVisionModel.h5')
font = cv2.FONT_HERSHEY_SIMPLEX
image3 = cv2.imread("trial.jpeg",0)
image3 = cv2.resize(image3,(256,256))
FRAME_WINDOW.image(image3)
# org
org = (80, 224)
fontScale = 1
color = (255, 0, 0)
thickness = 3
while run:
    return_value, frame1 = camera.read()
    #FRAME_WINDOW.image(frame1)
    time.sleep(0.001)

    return_value, frame2 = camera.read()
    #FRAME_WINDOW.image(frame2)

    time.sleep(0.001)
    cv2.imwrite("image1.png", frame1)
    image1 = cv2.imread("image1.png",0)
    #image1 = cv2.cvtColor(frame1, cv2.IMREAD_COLOR)
    #image1 = cv2.imdecode(frame1, cv2.IMREAD_COLOR)
    #image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1,(256,256))
    image1 = np.dstack([image1]*3)
    cv2.imwrite("image2.png", frame2)
    image2 = cv2.imread("image2.png",0)
    #image2 = cv2.cvtColor(frame2, cv2.IMREAD_COLOR)
    #image2 = cv2.imdecode(frame2, cv2.IMREAD_COLOR)
    #image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.resize(image2,(256,256))
    image2 = np.dstack([image2]*3)




    absdiff = cv2.absdiff(image1,image2)
    absdiff1 = np.expand_dims(absdiff, axis = 0)
    val = model2.predict(absdiff1)
    if val == 0:
         absdiff = cv2.putText(absdiff, 'Signed', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
         FRAME_WINDOW.image(absdiff)
    else:
         absdiff = cv2.putText(absdiff, 'Unsigned', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
         FRAME_WINDOW.image(absdiff)

else:
    st.markdown("<h4 style='text-align: center; color: gray;'>Bye..</h4>", unsafe_allow_html=True)"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Streamlit WebRTC Demo", page_icon="🤖")
task_list = ["Video Stream"]

with st.sidebar:
    st.title('Task Selection')
    task_name = st.selectbox("Select your tasks:", task_list)
st.title(task_name)

if task_name == task_list[0]:
    style_list = ['color', 'black and white']

    st.sidebar.header('Style Selection')
    style_selection = st.sidebar.selectbox("Choose your style:", style_list)

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model_lock = threading.Lock()
            self.style = style_list[0]

        def update_style(self, new_style):
            if self.style != new_style:
                with self.model_lock:
                    self.style = new_style

        def recv(self, frame):
            # img = frame.to_ndarray(format="bgr24")
            img = frame.to_image()
            if self.style == style_list[1]:
                img = img.convert("L")

            # return av.VideoFrame.from_ndarray(img, format="bgr24")
            return av.VideoFrame.from_image(img)

    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    if ctx.video_processor:
        ctx.video_transformer.update_style(style_selection)


    

    
   
