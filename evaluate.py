from train import CNN_RNN, device, test_loader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score

# Load the saved model
model = CNN_RNN()  # Use the correct model
model.load_state_dict(torch.load('sampleCNN_RNNv2.pth'))
model.to(device)
model.eval()

# Initialize variables for true and predicted labels
y_true = []
y_pred = []

# Evaluation
correct = 0
total = 0
accuracy_list = []  # Store accuracy values for each batch
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Append current batch of true and predicted labels
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        # Print progress
        if (i + 1) % 10 == 0:
            accuracy = 100 * correct / total
            print(f'Processed {i + 1} batches. Accuracy so far: {accuracy:.2f}%')
            accuracy_list.append(accuracy)  # Add current accuracy to the list

print('Test accuracy: %d %%' % (100 * correct / total))

# Calculate F1 score, precision and recall
f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
print('F1 Score: %.2f' % f1)
print('Precision: %.2f' % precision)
print('Recall: %.2f' % recall)

# Plotting
plt.figure(figsize=(10,5))
plt.title("Evaluation Accuracy")
plt.plot(accuracy_list, color='blue')  # Plot the list of accuracy values
plt.xlabel("Batches")
plt.ylabel("Accuracy")
plt.show()