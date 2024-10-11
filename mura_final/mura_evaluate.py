from sklearn.metrics import classification_report, cohen_kappa_score , accuracy_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, cohen_kappa_score , ConfusionMatrixDisplay
import seaborn as sns
from sklearn import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers, models
from keras.layers import concatenate, Dense, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout, \
    Attention
from keras import *
from keras.applications import MobileNet
from keras.applications import VGG16
from keras.applications.resnet import ResNet50
from keras import backend as K
from keras.layers import Layer
import keras.layers as kl
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
import keras.applications
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from my_functions import *
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input

target_categories = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
target_category = 'XR_WRIST'


class_mode = "binary"
loss_function = "binary_crossentropy"
num_classes = 2
batch_size = 8

#architecture="VGG"
test_dir = "C:/Users/raouf/Desktop/pfe/MURA-v1.1/valid_image_paths.csv"
df_test = load_data(test_dir, target_category)

from tensorflow.keras.preprocessing import image

path = "C:/Users/raouf/Desktop/pfe/history/new_arch/ensemble_all/XR_WRIST/ensemble_all_XR_WRIST.keras"


model = load_model(path , custom_objects={'F1Score': F1Score})

test_datagen = image.ImageDataGenerator()
test_generator = test_datagen.flow_from_dataframe(df_test,
                                                  directory=None,
                                                  x_col='image_path',
                                                  y_col='label',
                                                  target_size=(224,224),
                                                  batch_size=batch_size,
                                                  class_mode=class_mode,
                                                  shuffle=False
                                                  )

"""# Load the trained model

results = model.evaluate(test_generator)
print(results)
acc=int(results[5] * 100)
kap = int(results[-1] * 100)

import pyperclip
add = "_evaluate_acc_" + str(acc) +"_kappa_" + str(kap)

#pyperclip.copy(add)
"""


predictions = model.predict(test_generator)

#predicted_classes = np.argmax(predictions, axis=1)
predicted_classes = (predictions > 0.5).astype(int)


true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

report = classification_report(true_classes, predicted_classes, target_names=class_labels)

print(report)



# Generate confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
TN, FP, FN, TP = cm.ravel()
conf_matrix = np.array([[TN, FP],
                        [FN, TP]])

labels = ['Negative','Positive']

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
disp.plot(cmap='Blues')

# Customize the plot
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
# Increase the font size of the numbers inside the matrix
for texts in disp.text_.ravel():
    texts.set_fontsize(16)
plt.show()











accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes)
kappa = cohen_kappa_score(true_classes, predicted_classes)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print(f"Cohen's Kappa: {kappa}")


