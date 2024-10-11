import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from my_functions import *


model_path = "C:/Users/raouf/Desktop/pfe/history/final/densenet/XR_FINGER/best/finetuned_DenseNet_XR_FINGER.keras"
folder_path = "C:/Users/raouf/Desktop/pfe/new_mura/val/XR_FINGER/negative"


model = load_model(model_path, custom_objects={'F1Score': F1Score})  # Ensure custom object 'F1Score' is defined






def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img_array = np.expand_dims(img, axis=0)  # Expand dimensions to create a batch
    img_array = densenet_preprocess_input(img_array)
    return img_array


# Get a list of all files in the specified directory
files = [file for file in os.listdir(folder_path)]  # Filter image files


# Initialize counters
frac = 0
not_frac = 0
class_names = ["positive" , "negative" ]
threshold = 0.5
predictions=[]

# Iterate over each file
for i, file_name in enumerate(files):
    image_path = os.path.join(folder_path, file_name)
    test_image = preprocess_image(image_path)
    prediction_prob = model.predict(test_image)[0][0]

    predictions.append(prediction_prob)

    # Determine the result based on the threshold
    if prediction_prob < threshold:
        result = class_names[1]  # 'not fractured'
        not_frac += 1
    else:
        result = class_names[0]  # 'fractured'
        frac += 1

    # Print progress
    print(f"{round(100 * (i + 1) / len(files), 2)} % Done")

# Print final results
print(f"fractured     : {frac} occurrences, percentage: {100 * frac / len(files):.2f}%")
print(f"not fractured : {not_frac} occurrences, percentage: {100 * not_frac / len(files):.2f}%")


print("predictions \n" , predictions)
