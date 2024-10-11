import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Average
import seaborn as sns

from my_functions import *  # Assuming this contains some utility functions you need

mura_location = "C:/Users/raouf/Desktop/pfe/"

def load_data(file, part):
    train_data = pd.read_csv(file)
    train_data = train_data.rename(columns={train_data.columns[0]: 'image_path'})
    train_data['label'] = 2
    string_to_add = mura_location
    train_data['image_path'] = string_to_add + train_data['image_path'].astype(str)
    for index, row in train_data.iterrows():
        if 'positive' in str(row['image_path']).lower():
            train_data.at[index, 'label'] = "positive"
        else:
            train_data.at[index, 'label'] = "negative"

    if part == 0:
        pass
    else:
        train_data = train_data[train_data['image_path'].str.contains(part)]
    return train_data

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
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)

# Parameters
target_category = 'XR_HUMERUS'
class_mode = "binary"
batch_size = 8
test_dir = "C:/Users/raouf/Desktop/pfe/MURA-v1.1/valid_image_paths.csv"

# Load test data
df_test = load_data(test_dir, target_category)

# Load models
densenet_path = "C:/Users/raouf/Desktop/HUMERUS/DenseNet50_NoInclude_top_FC_1024_512_sigmoid_XR_HUMERUS.keras"
resnet_path = "C:/Users/raouf/Desktop/HUMERUS/humerus_ResNet50_and_Dense_sigmoid_XR_ELBOW.keras"
vgg_path = "C:/Users/raouf/Desktop/HUMERUS/Fine_tuning_vgg16_Dense256_sigmoid_XR_HUMERUS.keras"

# Ensure the models are loaded with custom objects
densenet_model = load_model(densenet_path)
resnet_model = load_model(resnet_path, custom_objects={"F1Score": F1Score})
vgg_model = load_model(vgg_path, custom_objects={"F1Score": F1Score})

# Create test data generator
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)
test_generator = test_datagen.flow_from_dataframe(df_test,
                                                  directory=None,
                                                  x_col='image_path',
                                                  y_col='label',
                                                  target_size=(224, 224),
                                                  batch_size=batch_size,
                                                  class_mode=class_mode,
                                                  shuffle=False)

# Evaluate and predict with each model
densenet_results = densenet_model.evaluate(test_generator)
densenet_predictions = densenet_model.predict(test_generator)

test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)
test_generator = test_datagen.flow_from_dataframe(df_test,
                                                  directory=None,
                                                  x_col='image_path',
                                                  y_col='label',
                                                  target_size=(224, 224),
                                                  batch_size=batch_size,
                                                  class_mode=class_mode,
                                                  shuffle=False)

resnet_results = resnet_model.evaluate(test_generator)
resnet_predictions = resnet_model.predict(test_generator)

test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
test_generator = test_datagen.flow_from_dataframe(df_test,
                                                  directory=None,
                                                  x_col='image_path',
                                                  y_col='label',
                                                  target_size=(224, 224),
                                                  batch_size=batch_size,
                                                  class_mode=class_mode,
                                                  shuffle=False)

vgg_results = vgg_model.evaluate(test_generator)
vgg_predictions = vgg_model.predict(test_generator)

# Average predictions
y_pred = np.array([densenet_predictions, resnet_predictions, vgg_predictions])
y_pred_avg = np.mean(y_pred, axis=0)
predicted_classes = (y_pred_avg > 0.5).astype(int)

# Evaluation
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Cohen's kappa score
kappa = cohen_kappa_score(true_classes, predicted_classes)
print("kappa:", kappa)

# Confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
cm_display.plot()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
