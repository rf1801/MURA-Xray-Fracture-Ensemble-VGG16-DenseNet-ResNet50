import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

myLine=5

# Define the folder containing the Excel files
folder_path = 'C:/Users/raouf/Desktop/pfe/history/keras_soft/'
# Get a list of all files in the specified directory
files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]

# Iterate over each file
for file_name in files:
    # Construct the full path to the file
    file_path = os.path.join(folder_path, file_name)

    # Read data from Excel file
    data = pd.read_excel(file_path)

    # Plotting
    fig, axs = plt.subplots(2, figsize=(12, 12))

    # Plot loss and validation loss
    axs[0].plot(range(1, len(data['loss']) + 1), data['loss'], label='Loss', color='blue', linewidth=myLine)
    axs[0].plot(range(1, len(data['val_loss']) + 1), data['val_loss'], label='Validation Loss', color='red', linewidth=myLine)

    # Set title and labels for loss plot
    axs[0].set_title(f'Loss Plot for {file_name}')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Set y-axis limits for loss plot
    axs[0].set_ylim(0.0, 1.0)

    # Set y-axis ticks and grid lines for loss plot
    axs[0].set_yticks(np.arange(0.0, 1.1, 0.1))
    axs[0].grid(True, which='both', axis='y', linestyle='--')

    # Set x-axis ticks and grid lines for loss plot
    axs[0].set_xticks(np.arange(1, len(data['loss']) + 1, 1))
    axs[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    axs[0].grid(True, which='both', axis='x', linestyle='--')

    # Plot accuracy and validation accuracy
    axs[1].plot(range(1, len(data['accuracy']) + 1), data['accuracy'], label='Accuracy', color='green', linewidth=myLine)
    axs[1].plot(range(1, len(data['val_accuracy']) + 1), data['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=myLine)

    # Set title and labels for accuracy plot
    axs[1].set_title(f'Accuracy Plot for {file_name}')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    # Set y-axis limits for accuracy plot
    axs[1].set_ylim(0.0, 1.0)

    # Set y-axis ticks and grid lines for accuracy plot
    axs[1].set_yticks(np.arange(0.0, 1.1, 0.1))
    axs[1].grid(True, which='both', axis='y', linestyle='--')

    # Set x-axis ticks and grid lines for accuracy plot
    axs[1].set_xticks(np.arange(1, len(data['accuracy']) + 1, 1))
    axs[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    axs[1].grid(True, which='both', axis='x', linestyle='--')

    # Show plot
    plt.show()
