from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import yaml
import os
import io
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to load PyTorch model weights from a YAML file
def load_yaml_weights(file_path):
    print(f"Attempting to load weights from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as yaml_file:
        weights = yaml.safe_load(yaml_file)
        print("YAML file loaded successfully.")
    
    state_dict = {}

    # Map the YAML weights to the model's layer names
    for key, value in weights.items():
        if 'weight' in key:  # This is to handle weights
            state_dict[key] = torch.tensor(value)
        elif 'bias' in key:  # This is to handle biases
            state_dict[key] = torch.tensor(value)

    print("Weights successfully converted to state_dict.")
    return state_dict

# Function to initialize and load a PyTorch ResNet18 model
def load_model(weights_path):
    model = models.resnet18(weights=None)  # Initialize without pretrained weights
    print("Model initialized successfully.")
    
    # Load the model weights
    print("Loading weights into the model...")
    state_dict = load_yaml_weights(weights_path)
    model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully.")
    
    # Set the model to evaluation mode
    model.eval()
    print("Model set to evaluation mode.")
    
    return model

# Define the weights file path
weights_path = "backend/model_weights.yaml"

# Load the model with the weights
model = load_model(weights_path)

# Define the image transformation pipeline (to match the input format for ResNet18)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

import traceback

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure an image file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # Check if the file is a valid image
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read the image
        img = Image.open(io.BytesIO(file.read()))
        
        # Apply the necessary transformations
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Make predictions
        with torch.no_grad():
            output = model(img_tensor)
        
        # Get the predicted class (index of highest score)
        _, predicted_class = torch.max(output, 1)
        
        # Return the prediction as a response
        return jsonify({'predicted_class': predicted_class.item()}), 200
    
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()  # Get the full traceback
        print(f"Error: {error_message}")
        print(f"Traceback: {error_traceback}")
        return jsonify({'error': error_message, 'traceback': error_traceback}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
