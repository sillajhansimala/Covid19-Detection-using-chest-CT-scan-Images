COVID-19 Detection Using Chest CT Scan Images

📌 Project Overview

This project aims to detect COVID-19 using chest CT scan images by leveraging deep learning techniques. I use a Convolutional Neural Network (CNN) model to classify images and provide predictions based on the input data.

🛠 Technologies Used

Python
Flask (Backend API)
OpenCV (Image Processing)
NumPy
PyTorch (Deep Learning Framework)
ResNet18 (Pre-trained CNN Model)
HTML, CSS, JavaScript (Frontend)

## 📂 Project Structure (Step-by-Step)

### 1. Backend Setup (`backend/`)
- **`app.py`**: This file contains the Flask backend code responsible for handling incoming requests and routing them to appropriate functions.
- **`load_model.py`**: This script loads and defines the CNN model (ResNet18) used for classification.
- **`preprocess.py`**: This file contains the code to preprocess input images, making them suitable for feeding into the model.
- **`model_weights.yaml`**: This file contains the pre-trained weights for the ResNet18 model.
- **`model.pkl`**: This serialized file contains the trained model that can be loaded for deployment.

### 2. Frontend Setup (`frontend/`)
- **`index.html`**: This is the main HTML file that defines the structure of the frontend user interface (UI).
- **`scripts.js`**: This JavaScript file contains the necessary code to handle the logic and user interactions for the frontend.
- **`styles.css`**: This file includes the CSS styles for designing the frontend to improve user experience.

### 3. Dataset Preparation (`dataset/`)
- **`images/`**: This folder contains the raw CT scan images used for training and testing the model.
- **`preprocessed_images/`**: This folder contains images that have been preprocessed, ready to be used for training or testing.
- **`split/`**: This folder contains images split into different datasets—`train/`, `test/`, and `val/` (validation) for model evaluation.

### 4. Project Documentation
- **`README.md`**: This file contains the documentation for the project, explaining the purpose, setup instructions, technologies used, and future improvements.

🧠 Model Used

I utilize ResNet18, a deep learning model, to classify chest CT scan images. The model takes preprocessed images as input and outputs a numerical prediction value.

📊 Prediction Output

The model provides a numerical output such as 733, 250, 323, etc. This value needs to be mapped to a meaningful classification, such as "COVID Positive" or "COVID Negative," in future updates.

🚀 Future Improvements

Implement a threshold-based classification to return "COVID Positive" or "COVID Negative."
Improve frontend UI for better user experience.
Enhance preprocessing techniques for better model accuracy.
