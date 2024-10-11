import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the folder containing the .keras model files
folder_path = "C:/Users/raouf/Desktop/pfe/history/keras_soft/"

# Get a list of all files in the specified directory ending with .keras
model_files = [file for file in os.listdir(folder_path) if file.endswith('.keras')]

# Define parameters for evaluation
test_dir = "C:/Users/raouf/Desktop/pfe/my_dataset/test"
batch_size = 64

# Load test data
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(224, 224),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)

# Iterate over each model file
for model_file in model_files:
    print(f"Evaluating model: {model_file}")

    # Load the trained model
    model_path = os.path.join(folder_path, model_file)
    model = load_model(model_path)

    # Make predictions on test data
    predictions = model.predict(test_generator)

    # Convert predictions to class labels
    predicted_classes = np.argmax(predictions, axis=1)

    # Get true labels from test data generator
    true_classes = test_generator.classes

    # Get class labels
    class_labels = list(test_generator.class_indices.keys())

    # Generate classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print("Classification Report:")
    print(report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)

    # Plot confusion matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
    cm_display.plot()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix '+model_file)
    plt.show()
