from sklearn.metrics import classification_report, cohen_kappa_score ,accuracy_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, cohen_kappa_score , ConfusionMatrixDisplay
import seaborn as sns
from sklearn import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers, models
from keras.layers import concatenate, Dense, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout, \
    Attention
from keras import *
import tensorflow_addons as tfa
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
from sklearn.metrics import cohen_kappa_score
import glob
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

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
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)




target_categories = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
target_category = 'XR_FINGER'

class_mode = "binary"
loss_function = "binary_crossentropy"
num_classes = 2
batch_size = 8


test_dir = "C:/Users/raouf/Desktop/pfe/MURA-v1.1/valid_image_paths.csv"

df_test = load_data(test_dir, target_category)



densenet_path = "C:/Users/raouf/Desktop/pfe/history/final/densenet/" + target_category + "/best/"
resnet_path = "C:/Users/raouf/Desktop/pfe/history/final/resnet/" + target_category + "/best/"
vgg_path = "C:/Users/raouf/Desktop/pfe/history/final/vgg/" + target_category + "/best/"




files = glob.glob(os.path.join(densenet_path, "*"))
densenet_path = files[0]
#densenet_path = "C:/Users/raouf/Desktop/HUMERUS/DenseNet50_NoInclude_top_FC_1024_512_sigmoid_XR_HUMERUS.keras"
densenet_model = load_model(densenet_path, custom_objects={'F1Score': F1Score})
test_datagen = image.ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)
test_generator = test_datagen.flow_from_dataframe(df_test,
                                                  directory=None,
                                                  x_col='image_path',
                                                  y_col='label',
                                                  target_size=(224,224),
                                                  batch_size=batch_size,
                                                  class_mode=class_mode,
                                                  shuffle=False
                                                  )
#results = densenet_model.evaluate(test_generator)

p1 = densenet_model.predict(test_generator)


files = glob.glob(os.path.join(resnet_path, "*"))
resnet_path = files[0]

#resnet_path="C:/Users/raouf/Desktop/HUMERUS/humerus_ResNet50_and_Dense_sigmoid_XR_ELBOW.keras"
resnet_model = load_model(resnet_path, custom_objects={"F1Score": F1Score})
test_datagen = image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)
test_generator = test_datagen.flow_from_dataframe(df_test,
                                                  directory=None,
                                                  x_col='image_path',
                                                  y_col='label',
                                                  target_size=(224,224),
                                                  batch_size=batch_size,
                                                  class_mode=class_mode,
                                                  shuffle=False
                                                  )
#results = resnet_model.evaluate(test_generator)

p2 = resnet_model.predict(test_generator)


files = glob.glob(os.path.join(vgg_path, "*"))
vgg_path = files[0]
#vgg_path= "C:/Users/raouf/Desktop/HUMERUS/Fine_tuning_vgg16_Dense256_sigmoid_XR_HUMERUS.keras"
vgg_model = load_model(vgg_path, custom_objects={"F1Score": F1Score})
test_datagen = image.ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
test_generator = test_datagen.flow_from_dataframe(df_test,
                                                  directory=None,
                                                  x_col='image_path',
                                                  y_col='label',
                                                  target_size=(224,224),
                                                  batch_size=batch_size,
                                                  class_mode=class_mode,
                                                  shuffle=False
                                                  )
#results = vgg_model.evaluate(test_generator)
p3 = vgg_model.predict(test_generator)








y_pred=[]
y_pred.append(p1)
y_pred.append(p2)
y_pred.append(p3)
from keras.layers import Average
y_pred_avg=np.array(Average()(y_pred))
predicted_classes = (y_pred_avg > 0.5).astype(int)


true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

report = classification_report(true_classes, predicted_classes, target_names=class_labels)

print(report)

kappa = cohen_kappa_score(true_classes, predicted_classes)
print("kappa ", kappa)

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

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print(f"Cohen's Kappa: {kappa}")