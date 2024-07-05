import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import requests
import os


# Load your trained model

#https://github.com/j-poddar/emergency-vehicle-detector/releases/download/saved_h5_model_v1/model_vgg16.h5
#model = load_model('model_vgg16.h5')


#model_url = "https://github.com/j-poddar/emergency-vehicle-detector/releases/download/saved_h5_model_v1/model_vgg16.h5"


# URL of the .h5 model file in the GitHub release
model_url = 'https://github.com/j-poddar/emergency-vehicle-detector/releases/download/saved_h5_model_v1/model_vgg16.h5'
model_filename = 'model_vgg16.h5'

# Download the model file
if not os.path.exists(model_filename):
    with st.spinner('Downloading model...'):
        response = requests.get(model_url, verify=False)
        with open(model_filename, 'wb') as f:
            f.write(response.content)

# Load your trained model
model = load_model(model_filename)


# Custom CSS for beautification
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .header {
        font-size:40px;
        color:#FF6347;
        text-align:center;
    }
    .subheader {
        font-size:20px;
        color:#4682B4;
        text-align:center;
    }
    </style>
    """, unsafe_allow_html=True)

# Header and Subheader
st.markdown('<p class="header">Emergency Vehicle Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload an image of a vehicle to determine if it\'s an emergency vehicle or not.</p>', unsafe_allow_html=True)

# Sidebar for additional information
st.sidebar.title("About")
st.sidebar.info(
    """
    This app uses a deep learning model to classify vehicles as emergency or non-emergency.
    Upload an image, and the model will predict the category.
    """
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Layout for displaying image and prediction result
if uploaded_file is not None:
    image = Image.open(uploaded_file)
   
    # Display image in columns for better layout
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        st.write("")
        #st.write("Classifying...")

        # Preprocess the image
        size = (224, 224)  # or the size your model expects
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img = img / 255.0  # normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # add batch dimension

        with st.spinner('Predicting...'):
            prediction = model.predict(img)
            if prediction[0][0] > 0.5:  # assuming the model outputs a single value, higher means emergency
                result = 'Emergency Vehicle'
                result_color = '#FF4500'  # Red color for emergency
            else:
                result = 'Non-emergency Vehicle'
                result_color = '#32CD32'  # Green color for non-emergency

        st.markdown(f'<p style="font-size:30px; color:{result_color};">{result}</p>', unsafe_allow_html=True)
