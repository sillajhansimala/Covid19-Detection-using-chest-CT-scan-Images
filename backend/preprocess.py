import os
import cv2
import numpy as np
from PIL import Image  # Using PIL to handle RGBA to RGB conversion

# Directory paths
input_dir = "dataset/images/"
output_dir = "dataset/preprocessed_images/"
os.makedirs(output_dir, exist_ok=True)

# Preprocessing parameters
target_size = (224, 224)  # Resize all images to 224x224 pixels
normalize = True          # Normalize pixel values to ImageNet standards

# ImageNet Normalization values (mean and std)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Preprocessing function
def preprocess_image(image_path, output_path):
    try:
        # Check if file is an image (not a directory)
        if not os.path.isfile(image_path):
            print(f"Skipping non-file: {image_path}")
            return
        
        # Open image using PIL to handle RGBA to RGB conversion
        image = Image.open(image_path)
        
        # Convert RGBA to RGB if the image has 4 channels (for normal images with alpha channel)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Convert image to numpy array
        image = np.array(image)
        
        # Handle grayscale (convert grayscale to RGB)
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA image
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize image to the target size
        image_resized = cv2.resize(image, target_size)
        
        # Normalize the pixel values (ImageNet standards)
        image_normalized = (image_resized / 255.0 - mean) / std
        
        # Save preprocessed image as PNG or JPEG
        output_file = os.path.join(output_path, os.path.basename(image_path))
        # Convert image to uint8 for saving (in case you want to save it as an image)
        image_to_save = ((image_normalized * 255).astype(np.uint8))
        cv2.imwrite(output_file, cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))  # Convert back to BGR before saving
        print(f"Processed: {image_path} -> {output_file}")
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Iterate through all images in the directory
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    preprocess_image(file_path, output_dir)

print("Preprocessing completed.")
