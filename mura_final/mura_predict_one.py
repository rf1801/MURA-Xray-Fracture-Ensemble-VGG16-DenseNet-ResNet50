import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model_path = "C:/Users/raouf/Desktop/pfe/history/no_overfit/mobilenet_epochs_3_batch_64_lr_0.001_attention_False.keras"
image_path = "C:/Users/raouf/Desktop/pfe/binary_mura/val/not fractured/your_image.jpg"  # Replace with the path to your image file

# Load the model
model = load_model(model_path)

# Load and preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create a batch
    return img_array

# Preprocess the image
test_image = preprocess_image(image_path)

# Get class names
class_names = ['fractured', 'not fractured']  # Replace with your actual class names

# Predict
prediction = model.predict(test_image)
predicted_class_index = np.argmax(prediction)
predicted_class = class_names[predicted_class_index]

# Output the prediction
print("Predicted class:", predicted_class)
