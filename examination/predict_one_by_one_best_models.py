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
    "Finger":  [global_path+"finetuned_DenseNet_XR_FINGER.keras",global_path+"finetuned_ResNet_XR_FINGER.keras", global_path+"finetuned_VGG_XR_FINGER.keras" ],
    "Forearm": ["Mobilenetmodel.keras"],
    "Wrist":["finetuned_DenseNet_XR_WRIST.keras"],
    "Humerus": ["Mobilenetmodel.keras"],
    "Shoulder": ["Mobilenetmodel.keras"]
}

body_part="Wrist"

models = model_files.get(body_part)



image_path = "C:/Users/raouf/Desktop/pfe/new_mura/val/XR_FINGER/positive/fingposv (4).png"
class_names = ["positive", "negative"]
threshold = 0.5



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





pcount=0
ncount=0


folder_path = "C:/Users/raouf/Desktop/pfe/new_mura/val/XR_WRIST/negative"
files = [file for file in os.listdir(folder_path)]
for image_path in files:
    image_path = os.path.join(folder_path, image_path)
    predictions = []
    for model_file in models:
        model = load_model(global_path+model_file, custom_objects={'F1Score': F1Score})
        test_image = preprocess_image(image_path, model_file)
        prediction_prob = model.predict(test_image)[0][0]
        predictions.append(prediction_prob)
    pred_avg = np.array(Average()(predictions))
    prediction_prob = (pred_avg > 0.5).astype(int)
    if prediction_prob < threshold:
        result = class_names[1]
        ncount = ncount + 1
    else:
        result = class_names[0]
        pcount = pcount + 1



#print(prediction_prob)
#print(result)
print("pos = ",pcount)
print("neg = ",ncount)