import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

dataset_path = "C/Users/lakshay pandey/Desktop/AI DA/dataset/Indian"

def load_data(dataset_path, img_size=(28, 28)):
    data = []
    labels = []

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        if folder_name.isdigit():  # Numeric folder name (1-9)
            label = int(folder_name) - 1  # Map 1 to 0, 2 to 1, ..., 9 to 8
        elif folder_name.isalpha() and len(folder_name) == 1:  # Alphabetic folder name (A-Z)
            label = ord(folder_name.upper()) - ord('A') + 9  # Map A to 9, B to 10, ..., Z to 34
        else:
            print(f"Warning: Unexpected folder name '{folder_name}'. Skipping.")
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Warning: Unable to read image {img_path}. Skipping.")
                continue

            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0
            data.append(img)
            labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    data = data.reshape((data.shape[0], img_size[0], img_size[1], 1))

    num_classes = 35
    labels = to_categorical(labels, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
