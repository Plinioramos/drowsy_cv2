# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:38:49 2024

@author: plini
"""

import streamlit as st
import cv2
import numpy as np
import os
import time
import dlib
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Funções para cada página
def page_users_guide():
    st.title("User's Guide")
    
    st.markdown("""
    ## Introduction
    Welcome to **DrowsinessNET**! This guide will help you understand how to use the application effectively to detect drowsiness using various methods.
    """)
    
    st.markdown("""
    ## Upload Your Data
    You can upload your data using one of the following methods:
    
    ### Option 1: Upload Data from Camera
    1. Go to the **Detect Drowsiness in Real Time** page.
    2. Select **Webcam** as the device.
    3. Click **Run** to start capturing data from your webcam.
    
    ### Option 2: Upload Data from Files
    1. Go to the **Offline Analysis** page.
    2. Select the file type you want to upload (Video or EDF file).
    3. Click **Upload** and choose your file.
    
    """)
    #![Upload Data](https://canto-wp-media.s3.amazonaws.com/app/uploads/2019/08/19192233/mp4-video-file-25-768x768.jpg)  colocar na parte do texto em cima
    
    #st.image("local_path_to_your_image.jpg", caption="Upload Data")  # Substitua pelo caminho relativo da sua imagem local
    
    
    st.markdown("""
    ## Detect Drowsiness in Real Time
    Detect drowsiness using live data from your webcam or an EEG headset.
    
    ### Using Webcam
    1. Go to the **Detect Drowsiness in Real Time** page.
    2. Choose **Webcam** as the device.
    3. Click **Run** to start real-time drowsiness detection.
    
    ![Webcam Detection](url-to-screenshot)
    
    ### Using EEG Headset
    1. Go to the **Detect Drowsiness in Real Time** page.
    2. Choose **EEG Device** as the device.
    3. Follow the instructions to connect your EEG headset.
    """)
    
    st.markdown("""
    ## Offline Analysis
    Analyze pre-recorded data to detect drowsiness.
    
    ### Use Default Model
    1. Go to the **Offline Analysis** page.
    2. Choose **Use default model**.
    3. Upload your video or EDF file.
    4. Click **Proceed** to analyze the data.
    
    ![Offline Analysis](url-to-screenshot)
    
    ### Use User-Saved Model
    1. Go to the **Offline Analysis** page.
    2. Choose **Use user-saved model**.
    3. Upload your trained video or EEG model file.
    4. Upload your video or EDF file.
    5. Click **Proceed** to analyze the data.
    """)
    
    st.markdown("""
    ## Training the Model
    Train the drowsiness detection model with your own data.
    
    ### Upload Training Data
    1. Go to the **Train the model** section in the **Offline Analysis** page.
    2. Upload your video or EDF file for training.
    3. Upload your video or EDF file for testing.
    4. Click **Proceed** to start training.
    
    ![Training Model](url-to-screenshot)
    """)
    
    st.markdown("""
    ## Tips and Recommendations
    - Ensure good lighting conditions when using the webcam for better accuracy.
    - Make sure your EEG headset is properly calibrated and connected.
    - Use high-quality video files for offline analysis.
    - Follow the instructions carefully to avoid errors.
    """)
    
    st.markdown("""
    ## FAQs
    **Q: What should I do if the webcam is not working?**
    - A: Check if the webcam is properly connected and try restarting the application.
    
    **Q: Can I use other file formats for uploading data?**
    - A: Currently, only MP4, AVI, and EDF files are supported.
    
    **Q: How can I improve the accuracy of drowsiness detection?**
    - A: Ensure good lighting and clear visibility of the face when using the webcam. For EEG, make sure the headset is properly calibrated.
    """)

    st.markdown("""
    ### We hope this guide helps you use the application effectively. For further assistance, please contact our support team.
    """)

def page_offline_analysis():
    st.title("Offline Analysis")
    option = st.radio("Choose an option:", ["Use default model", "Use user-saved model", "Train the model"])
    
    if option == "Use default model":
        with st.expander("Upload video or EDF file to test"):
            st.write("Upload video or EDF file to test")
        col1, col2 = st.columns(2)
        with col1:
            video_file = st.file_uploader("Upload Video File", type=["mp4", "avi"])
        with col2:
            edf_file = st.file_uploader("Upload EDF File", type=["edf"])
    
    elif option == "Use user-saved model":
        with st.expander("Upload trained video or EEG model"):
            st.write("Upload trained video or EEG model")
        video_model_file = st.file_uploader("Upload Video Model File", type=["h5", "pkl"])
        eeg_model_file = st.file_uploader("Upload EEG Model File", type=["h5", "pkl"])
        st.write("Upload video or EDF file to test")
        col1, col2 = st.columns(2)
        with col1:
            video_file = st.file_uploader("Upload Video File", type=["mp4", "avi"])
        with col2:
            edf_file = st.file_uploader("Upload EDF File", type=["edf"])
    
    elif option == "Train the model":
        with st.expander("Upload video or EDF file to train"):
            st.write("Upload video or EDF file to train")
        video_file_train = st.file_uploader("Upload Video File for Training", type=["mp4", "avi"])
        edf_file_train = st.file_uploader("Upload EDF File for Training", type=["edf"])
        st.write("Upload video or EDF file to test")
        col1, col2 = st.columns(2)
        with col1:
            video_file = st.file_uploader("Upload Video File", type=["mp4", "avi"])
        with col2:
            edf_file = st.file_uploader("Upload EDF File", type=["edf"])
    
    # Handle the uploaded files
    if st.button("Proceed"):
        if video_file:
            st.write("Video file uploaded successfully.")
            # Process the video file
        if edf_file:
            st.write("EDF file uploaded successfully.")
            # Process the EDF file
        if video_model_file:
            st.write("Video model file uploaded successfully.")
            # Process the video model file
        if eeg_model_file:
            st.write("EEG model file uploaded successfully.")
            # Process the EEG model file

def page_detect_drowsiness():
    st.title("Detect Drowsiness in Real Time")
    option = st.radio("Choose an option:", ["Use default model", "Use user-saved model", "Train the model"])
    
    if option == "Use default model":
        with st.expander("Upload trained video or EEG model"):
            st.write("Upload trained video or EEG model")
        video_model_file = st.file_uploader("Upload Video Model File", type=["h5", "pkl"])
        eeg_model_file = st.file_uploader("Upload EEG Model File", type=["h5", "pkl"])
        st.write("Real-time testing")
        device = st.selectbox("Choose device:", ["Webcam", "EEG Device"])
        if device == "Webcam":
            run_webcam()

    elif option == "Use user-saved model":
        with st.expander("Use user-saved model"):
            st.write("Upload trained video or EEG model")
        video_model_file = st.file_uploader("Upload Video Model File", type=["h5", "pkl"])
        eeg_model_file = st.file_uploader("Upload EEG Model File", type=["h5", "pkl"])
        st.write("Real-time testing")
        device = st.selectbox("Choose device:", ["Webcam", "EEG Device"])
        if device == "Webcam":
            run_webcam()
    
    elif option == "Train the model":
        with st.expander("Training the Model"):
            st.write("Upload video or EDF file to train")
        file_type = st.selectbox("Choose file type:", ["Video file", "EDF file"])
        st.file_uploader(f"Upload {file_type} for training", type=["mp4", "avi", "edf"])
        st.write("Real-time testing")
        device = st.selectbox("Choose device:", ["Webcam", "EEG Device"])
        if device == "Webcam":
            run_webcam()



class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.crop_eyes = False
        self.duration = 30
        self.start_time = time.time()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.output_dir = "output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.crop_eyes:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            faces = self.detector(gray)
            for face in faces:
                shape = self.predictor(gray, face)
                x1 = shape.part(37).x
                x2 = shape.part(46).x
                y1 = shape.part(44).y
                y2 = shape.part(45).y
                crop_img = gray[y1-10:y2+10, x1-10:x2+10]
                file_name = f"{self.output_dir}/{self.count}.jpg"
                cv2.imwrite(file_name, crop_img)
                self.count += 1

        if time.time() - self.start_time > self.duration:
            st.stop()
        
        return av.VideoFrame.from_ndarray(img_rgb, format="rgb24")

def run_webcam():
    st.write("Starting webcam...")
    run = st.checkbox('Run')
    crop_eyes = st.checkbox('Crop Eyes')
    duration = st.number_input('Duration (seconds)', min_value=1, max_value=300, value=30)

    if run:
        webrtc_ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.crop_eyes = crop_eyes
            webrtc_ctx.video_processor.duration = duration

# Outras funções do código permanecem iguais

def main():
    st.sidebar.title("Navigation")
    
    #st.sidebar.image(r"C:\Users\plini\OneDrive\Documentos\Doutorado\TESE\Drowsy APP\drowsy.gif", use_column_width=True)  # Substitua pelo URL do seu GIF

    if "page" not in st.session_state:
        st.session_state.page = "Main Page"

    if st.sidebar.button("Main Page"):
        st.session_state.page = "Main Page"
    if st.sidebar.button("User's Guide"):
        st.session_state.page = "User's Guide"
    if st.sidebar.button("Offline Analysis"):
        st.session_state.page = "Offline Analysis"
    if st.sidebar.button("Detect Drowsiness in Real Time"):
        st.session_state.page = "Detect Drowsiness in Real Time"

    page = st.session_state.page

    if page == "Main Page":
        st.title("DrowsinessNET")
        st.markdown("<h3 style='color: grey; font-family: Verdana, sans-serif; text-align: center;'>Central de Ajuda</h3>", unsafe_allow_html=True)
        st.image("drowsiness_net_image.png", use_column_width=True)
        st.markdown("<h4 style='text-align: center;'>What is DrowsinessNET?</h4>", unsafe_allow_html=True)
        st.write("DrowsinessNET is a system designed to detect drowsiness in individuals using various techniques such as image processing and EEG analysis.")
    elif page == "User's Guide":
        page_users_guide()
    elif page == "Offline Analysis":
        page_offline_analysis()
    elif page == "Detect Drowsiness in Real Time":
        page_detect_drowsiness()

if __name__ == "__main__":
    main()
