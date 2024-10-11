import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the folder location
folder_location = 'C:/Users/raouf/Desktop/pfe/history/compare/'

# Get a list of all files in the specified directory
files = [file for file in os.listdir(folder_location) if file.endswith('.xlsx')]

# Initialize a list to store validation accuracy data
val_accuracy_data = []

# Iterate over each file
for file in files:

    # Construct the full path to the file
    file_path = os.path.join(folder_location, file)

    try:
        # Read data from Excel file
        data = pd.read_excel(file_path)

        column_to_plot = "val_accuracy"

        # Extract validation accuracy
        val_accuracy = data[column_to_plot]

        # Append validation accuracy to the list along with the file name
        val_accuracy_data.append((file, val_accuracy))

    except Exception as e:
        # Print the filename if an error occurs during processing
        print(f"Error processing file: {file} - {e}")

# Plotting
plt.figure(figsize=(12, 6))

# Plot validation accuracy for each file
for file, val_accuracy in val_accuracy_data:
    x = np.array(range(len(val_accuracy)))
    y = np.array(val_accuracy)
    plt.plot(x, y, label=file, linewidth=2.5)

# Set labels and title
plt.xlabel('Epochs', fontsize='xx-large')
plt.ylabel(column_to_plot, fontsize='xx-large')
plt.title('Comparison', fontsize='xx-large')
plt.legend(fontsize='xx-small')

# Set y-axis limits and ticks to focus on the range of 0.3 to 0.9
plt.ylim(0.35, 0.9)
plt.xlim(0, 20)
plt.yticks(np.arange(0.35, 0.95, 0.05), fontsize='xx-large')
plt.xticks(fontsize='xx-large')

# Show grid
plt.grid(True)

# Show plot
plt.show()
