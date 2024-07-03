# emergency-vehicle-detector

![Emergency Vehicle Detector](https://github.com/j-poddar/emergency-vehicle-detector/blob/main/images/emergency_vehicle_detector_home_page.PNG)



## Overview

The Emergency Vehicle Detector is a deep learning-based application that classifies vehicles as either emergency or non-emergency. The model has been trained and deployed in the cloud, and this repository contains all the files necessary to run the application locally using Streamlit.


## Features

- Image Upload: Users can upload an image of a vehicle. The users can use any of the [sample test images](https://github.com/j-poddar/emergency-vehicle-detector/tree/main/images/sample_test_images) available in the repo.
- Prediction: The app predicts if the vehicle is an emergency vehicle or not.
- Real-time Results: The prediction is displayed along with the uploaded image.

## Live Demo

You can try the live demo of the application [here](https://emergency-vehicle-detector.streamlit.app).


## Model

The model is a Convolutional Neural Network (CNN) trained to classify images of vehicles. It is stored in the model_vgg16.h5 file stored in the 'Releases'.

## Acknowledgements

- This project uses [Streamlit](https://streamlit.io) for the web interface.
- The deep learning model is built with [TensorFlow](https://www.tensorflow.org).
