
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import layers, models
from keras.layers import concatenate, Dense, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout, Attention
from keras import *
from keras.applications import MobileNet
from keras.applications import VGG16
from keras.applications.resnet import ResNet50
from keras import backend as K
from keras.layers import Layer
import keras.layers as kl
#! pip install pandas
import pandas as pd
from keras.applications import EfficientNetB0
import keras.applications
import numpy as np
import os
import tensorflow_addons as tfa

class F1Score(tf.keras.metrics.Metric):
    """
    The tfa.metrics.F1Score (https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score)
    requires some reshaping that is inconsistent with the other metrics we like to track
    so we will define it from scratch.
    """
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

#target_categories = ['XR_FOREARM', 'XR_HUMERUS', 'XR_HAND', 'XR_FINGER',
#                     'XR_SHOULDER', 'XR_ELBOW', 'XR_WRIST']
target_category = 'XR_HUMERUS'
batch_size = 16
num_epochs = 100
validation_split = 0.2
loss_function = tf.keras.losses.BinaryCrossentropy()
learning_rate = 0.0001
input_shape = (224, 224, 3)
architecture = 'Ensemble_ResNet50_DenseNet_'
model_name = architecture+target_category
include_top=False
#metrics=["accuracy"] #, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
early_stopping_patience = 5
shuffle = True
class_mode = 'binary'#'categorical'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
trainable = False
training = False
dropout = 0.2
path_train_valid = "/media/mousser/HDD/wafa/"
METRICS = [
    tf.keras.metrics.TruePositives(name="tp"),
    tf.keras.metrics.FalsePositives(name="fp"),
    tf.keras.metrics.TrueNegatives(name="tn"),
    tf.keras.metrics.FalseNegatives(name="fn"),
    tf.keras.metrics.BinaryAccuracy(name="binary_acc"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    F1Score(name="f1_score"),
    tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
    tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    tfa.metrics.CohenKappa(name="cohen_kappa", num_classes=2),
]
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


test_dir = "C:/Users/raouf/Desktop/pfe/MURA-v1.1/valid_image_paths.csv"

df_test = load_data(test_dir, target_category)

# Define the ImageDataGenerator without augmentation for training data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30,
                                                          horizontal_flip=True,
                                                          fill_mode="constant",
                                                          cval=0.0)




# Define the ImageDataGenerator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
print('Test :')
# Generate test set from dataframe
test_generator = test_datagen.flow_from_dataframe(df_test,
                                             directory=None,
                                             x_col='image_path',
                                             y_col='label',
                                             target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             batch_size=batch_size, shuffle=False,
                                             class_mode=class_mode
                                             )

# Load finetuned densenet
densenet_model_name = "C:/Users/raouf/Desktop/HUMERUS/DenseNet50_NoInclude_top_FC_1024_512_sigmoid_XR_HUMERUS.keras"
densenet_model = tf.keras.models.load_model(densenet_model_name,
                                           custom_objects={"F1Score":F1Score})

# Load finetuned ResNet
resnet_model_name = "suffle=true+trainable_humerus_ResNet50_and_Dense_sigmoid_XR_ELBOW_with_fixed_train_valid_test_is_valid_rep.keras"
resnet_model = tf.keras.models.load_model(resnet_model_name,
                                          custom_objects={"F1Score":F1Score})

# Load finetuned vgg16
vgg_model_name = "Fine tuning_XR_FOREARM_vgg16_and_Dense256_sigmoid_XR_FOREARM_with_fixed_train_valid_test_is_valid_rep.keras"
vgg_model = tf.keras.models.load_model(vgg_model_name,
                                          custom_objects={"F1Score":F1Score})

from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet

y_pred=[]

# prediction using resnet
# Define the ImageDataGenerator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
print('Test :')
# Generate test set from dataframe
test_generator = test_datagen.flow_from_dataframe(df_test,
                                             directory=None,
                                             x_col='image_path',
                                             y_col='label',
                                             target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             batch_size=batch_size,
                                             shuffle=False,
                                             class_mode=class_mode
                                             )
# Make predictions using the test generator
y_pred_prob = resnet_model.predict(test_generator)
y_pred.append(y_pred_prob)

# prediction using densenet
# Define the ImageDataGenerator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)
print('Test :')
# Generate test set from dataframe
test_generator = test_datagen.flow_from_dataframe(df_test,
                                             directory=None,
                                             x_col='image_path',
                                             y_col='label',
                                             target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             batch_size=batch_size,
                                             shuffle=False,
                                             class_mode=class_mode
                                             )
# Make predictions using the test generator
y_pred_prob = densenet_model.predict(test_generator)
y_pred.append(y_pred_prob)

# prediction using vgg
# Define the ImageDataGenerator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
print('Test :')
# Generate test set from dataframe
test_generator = test_datagen.flow_from_dataframe(df_test,
                                             directory=None,
                                             x_col='image_path',
                                             y_col='label',
                                             target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             batch_size=batch_size,
                                             shuffle=False,
                                             class_mode=class_mode
                                             )
# Make predictions using the test generator
y_pred_prob = vgg_model.predict(test_generator)
y_pred.append(y_pred_prob)

from keras.layers import Average
# make average prediction
y_pred_avg=np.array(Average()(y_pred))
type(y_pred_avg)

# Convert predicted probabilities to binary predictions (0 or 1)
#y_pred = np.argmax(y_pred_prob, axis=1)
y_pred = (y_pred_avg > 0.5).astype(int)

from sklearn.metrics import classification_report
y_test = test_generator.classes
# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(y_test, y_pred)

#! pip install seaborn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

