import torch
import yaml
from torchvision import models
import os

# Print current directory files for debugging
print("Files in backend directory:", os.listdir("."))

# Initialize the ResNet18 model
print("Initializing ResNet18 model...")
model = models.resnet18(weights=None)  # Use `weights=None` to avoid using pretrained weights
print("Model initialized successfully.")

# Function to load weights from YAML and convert to state_dict
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
        # Ensure the key names in the YAML match those of the model's layers
        if 'weight' in key:  # This is to handle weights
            state_dict[key] = torch.tensor(value)
        elif 'bias' in key:  # This is to handle biases
            state_dict[key] = torch.tensor(value)

    print("Weights successfully converted to state_dict.")
    return state_dict

# Specify the weights file path
weights_path = "backend/model_weights.yaml"

# Load the weights
print("Checking weights file path:", weights_path)
state_dict = load_yaml_weights(weights_path)

# Load weights into the model
print("Loading weights into the model...")
try:
    model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Error loading state dict: {e}")

# Ensure the model is in evaluation mode
model.eval()
print("Model set to evaluation mode.")

# Test the model with a dummy input
print("Testing the model with a dummy input...")
dummy_input = torch.randn(1, 3, 224, 224)  # Example input for ResNet18
output = model(dummy_input)

# Print the shape of the output for clarity
print("Dummy input passed through the model successfully.")
print("Model output shape:", output.shape)
print("Model output (truncated):", output[:, :5])  # Print a truncated output for readability
