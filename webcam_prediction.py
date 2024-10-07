import os
import numpy as np
import cv2
from keras.models import load_model
from keras.optimizers import SGD

# Load the trained model
model = load_model('ISL_CNN_Model.h5')

# Recompile the model
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

# Create a mapping from numeric classes to alphabets and digits
class_mapping = {i: chr(i + 65) for i in range(26)}  # A-Z
class_mapping.update({i + 26: str(i + 1) for i in range(9)})  # 1-9

# Prediction function
def keras_predict(model, image):
    image = image.astype('float32') / 255.0  # Normalize the image
    data = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    pred_probab = model.predict(data)[0]  # Get prediction probabilities
    pred_class = np.argmax(pred_probab)  # Get the predicted class
    return max(pred_probab), pred_class

# Function to crop the image
def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

# Function to capture from webcam and predict
def capture_and_predict():
    cam_capture = cv2.VideoCapture(0)  # Start webcam
    while True:
        _, image_frame = cam_capture.read()
        if image_frame is None:
            break  # Break the loop if there is no frame

        # Crop the image (adjust coordinates as necessary)
        im2 = crop_image(image_frame, 300, 300, 300, 300)

        # Convert to grayscale
        image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        # Resize the image to 28x28
        img = cv2.resize(image_grayscale, (28, 28))

        # Predict the class
        probab, pred_class = keras_predict(model, img)

        # Map numeric class to alphabet or digit
        predicted_letter = class_mapping[pred_class]

        # Display the prediction
        cv2.putText(image_frame, f'Predicted: {predicted_letter}, Confidence: {probab:.2f}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("Webcam", image_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cam_capture.release()
    cv2.destroyAllWindows()

# Start capturing and predicting
capture_and_predict()
