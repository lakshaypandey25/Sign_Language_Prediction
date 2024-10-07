import os
import numpy as np
from keras.models import load_model
import cv2

# Load the trained model
model = load_model('ISL_CNN_Model.h5')

# Prediction function
def keras_predict(model, image):
    image = image.astype('float32') / 255.0  # Normalize the image
    data = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    pred_probab = model.predict(data)[0]  # Get prediction probabilities
    pred_class = np.argmax(pred_probab)  # Get the predicted class
    return max(pred_probab), pred_class

# Create a mapping from numeric classes to alphabets and digits
class_mapping = {i: chr(i + 65) for i in range(26)}  # A-Z
class_mapping.update({i + 26: str(i + 1) for i in range(9)})  # 1-9

# Directory containing subfolders for each letter
base_dir = "C:/Users/lakshay pandey/Desktop/AI DA/dataset/Indian"  # Updated path to dataset

# Loop through each letter folder
for letter in os.listdir(base_dir):
    letter_path = os.path.join(base_dir, letter)
    
    if os.path.isdir(letter_path):  # Check if it's a directory
        print(f"Processing folder: {letter}")
        
        # Loop through each image in the letter folder
        for filename in os.listdir(letter_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Check if the file is an image
                img_path = os.path.join(letter_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"Error: Unable to load image {filename}. Skipping...")
                else:
                    # Process the image
                    img = cv2.resize(img, (28, 28))  # Resize to 28x28 pixels
                    img = np.reshape(img, (28, 28, 1))  # Reshape for the model input

                    # Predict the class
                    probab, pred_class = keras_predict(model, img)

                    # Map numeric class to letter or digit
                    predicted_char = class_mapping[pred_class]

                    # Print result
                    print(f"Image: {filename}, Predicted Class: {predicted_char}, Confidence: {probab:.2f}")
            else:
                print(f"Skipping non-image file: {filename}")
    else:
        print(f"Skipping non-directory item: {letter}")
