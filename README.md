COVID-19 Detection Using Chest CT Scan Images

Project Overview

This project aims to detect COVID-19 using chest CT scan images by leveraging deep learning techniques. We use a Convolutional Neural Network (CNN) model to classify images and provide predictions based on the input data.

Technologies Used

Python

Flask (Backend API)

OpenCV (Image Processing)

NumPy

PyTorch (Deep Learning Framework)

ResNet18 (Pre-trained CNN Model)

HTML, CSS, JavaScript (Frontend)

Project Structure

Covid19-detection-using-chest-images/
│-- backend/
│   ├── app.py  # Flask backend to handle requests
│   ├── model.py  # Loads and defines the CNN model
│   ├── preprocess.py  # Preprocesses input images
│   ├── model_weights.yaml  # Model weights for ResNet18
│   ├── requirements.txt  # Dependencies
│-- frontend/
│   ├── index.html  # Frontend UI
│   ├── scripts.js  # JavaScript functionality
│   ├── styles.css  # Styling for the frontend
│-- dataset/
│   ├── images/  # Raw CT scan images
│   ├── preprocessed_images/  # Processed images
│-- README.md  # Project Documentation

Model Used

We utilize ResNet18, a deep learning model, to classify chest CT scan images. The model takes preprocessed images as input and outputs a numerical prediction value.

Prediction Output

The model provides a numerical output such as 733, 250, 323, etc. This value needs to be mapped to a meaningful classification, such as "COVID Positive" or "COVID Negative," in future updates.

Future Improvements

Implement a threshold-based classification to return "COVID Positive" or "COVID Negative."

Improve frontend UI for better user experience.

Enhance preprocessing techniques for better model accuracy.
