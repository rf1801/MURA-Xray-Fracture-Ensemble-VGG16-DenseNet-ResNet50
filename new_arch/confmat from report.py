import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Provided confusion matrix values

TN = 121
FP = 27
FN = 17
TP = 123



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