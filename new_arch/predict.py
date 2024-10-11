import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from my_functions2 import *


model_path = "C:/Users/raouf/Desktop/pfe/history/final/densenet/XR_WRIST/best/finetuned_DenseNet_XR_WRIST.keras"

image_path="C:/Users/raouf/Desktop/pfe/new_mura/val/XR_WRIST/positive/wriposv (52).png"

model = load_model(model_path, custom_objects={'F1Score': F1Score})




img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)






prediction_prob=model.predict(img_array)

print(prediction_prob)