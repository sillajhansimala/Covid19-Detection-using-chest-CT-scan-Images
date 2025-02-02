# Covid19-Detection-using-chest-CT-scan-Images
# COVID-19 Detection Using CT Scan Images

## 📌 Overview
This project aims to develop a deep learning-based model for detecting COVID-19 from CT scan images. The system leverages convolutional neural networks (CNNs) to classify CT images into COVID-19 positive and negative cases.

## ✨ Features
- ✅ Automatic detection of COVID-19 using CT scan images
- ✅ Deep learning-based classification using CNNs
- ✅ Preprocessing techniques to enhance image quality
- ✅ Performance evaluation using accuracy, precision, recall, and F1-score

## 📂 Dataset
The dataset consists of labeled CT scan images, categorized as:
- 🦠 COVID-19 Positive
- ❌ COVID-19 Negative

Ensure that the dataset is properly preprocessed before training.

## 🛠 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/covid19-detection-ct.git
   cd covid19-detection-ct
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage
1. Prepare the dataset and place it in the `data` directory.
2. Train the model:
   ```bash
   python train.py
   ```
3. Evaluate the model:
   ```bash
   python evaluate.py
   ```
4. Perform inference on new images:
   ```bash
   python predict.py --image path/to/image
   ```

## 🏗 Model Architecture
The model uses a CNN-based architecture with multiple convolutional layers, batch normalization, and dropout to prevent overfitting.

## 📊 Results
The model achieves high accuracy in detecting COVID-19 from CT images. Evaluation metrics include:
- 📈 Accuracy
- 🎯 Precision
- 🔁 Recall
- 📉 F1-score

## 🤝 Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## 📜 License
This project is licensed under the MIT License.

## 🙏 Acknowledgments
- 🏥 Researchers who provided the CT scan datasets
- 🛠 Open-source deep learning frameworks used in the project

