from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.layers import Average
import os

global_path="C:/Users/raouf/Desktop/models/"
model_files = {
    "Elbow": ["Mobilenetmodel.keras"],
    "Finger":  ["finetuned_DenseNet_XR_FINGER.keras","finetuned_ResNet_XR_FINGER.keras", "finetuned_VGG_XR_FINGER.keras" ],
    "Forearm": ["Mobilenetmodel.keras"],
    "Wrist":["finetuned_DenseNet_XR_WRIST.keras"],
    "Humerus": ["Mobilenetmodel.keras"],
    "Shoulder": ["finetuned_ResNet_XR_SHOULDER.keras","finetuned_DenseNet_XR_SHOULDER.keras","finetuned_VGG_XR_SHOULDER.keras"]
}

body_part="Shoulder"
models = model_files.get(body_part)

class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name="f1", initializer="zeros")
        self.precision_fn = tf.keras.metrics.Precision()
        self.recall_fn = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        self.f1.assign(2 * ((p * r) / (p + r + 1e-10)))

    def result(self):
        return self.f1

    def reset_state(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)


def preprocess_image(image_path, model_name , target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img_array = np.expand_dims(img, axis=0)
    if "Dense" in model_name:
        img_array = densenet_preprocess_input(img_array)
    elif "Res" in model_name:
        img_array = resnet50_preprocess_input(img_array)
    elif "VGG" in model_name:
        img_array = vgg16_preprocess_input(img_array)
    return img_array


file_path="C:/Users/raouf/Desktop/pfe/new_mura/val/XR_SHOULDER/positive/shouldposv (32).png"

predictions = []
for model_file in models:
    model = load_model(global_path+model_file, custom_objects={'F1Score': F1Score})
    test_image = preprocess_image(file_path, model_file)
    prediction_prob = model.predict(test_image)[0][0]
    predictions.append(prediction_prob)
pred_avg = np.array(Average()(predictions))

prob_fractured = pred_avg
prob_not_fractured = (1 - prob_fractured)
if prob_fractured >= 0.5:
    prediction = "Fractured"
else:
    prediction = "Not Fractured"

print(prob_fractured)
print(prob_not_fractured)
