import pandas as pd
import numpy as np
import librosa
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from torch import flatten
from torch.nn import functional as F
import pickle

import os
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device:", device)

# Load the processed data
x_train = np.load('processedDataSample/train_data_X.npy')
y_train = np.load('processedDataSample/train_data_y.npy')
x_val = np.load('processedDataSample/val_data_X.npy')
y_val = np.load('processedDataSample/val_data_y.npy')
x_test = np.load('processedDataSample/test_data_X.npy')
y_test = np.load('processedDataSample/test_data_y.npy')

print("Data loaded successfully")

# Convert data to PyTorch tensors and create dataloaders
train_data = TensorDataset(torch.from_numpy(x_train.reshape(-1, 1, x_train.shape[1], 1)), torch.from_numpy(y_train))
val_data = TensorDataset(torch.from_numpy(x_val.reshape(-1, 1, x_val.shape[1], 1)), torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(x_test.reshape(-1, 1, x_test.shape[1], 1)), torch.from_numpy(y_test))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

print("Data loaders created")

# Define the CNN model
class CNN_RNN(nn.Module):
    def __init__(self):
        super(CNN_RNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.pool = nn.MaxPool2d((2, 1), 2)
        self.rnn = nn.RNN(32 * 20 * 1, 64, batch_first=True)  # RNN layer
        self.fc1 = nn.Linear(64, 128)  # Adjusted input features
        self.fc2 = nn.Linear(128, 10)  # Adjusted output features

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten the output
        x, _ = self.rnn(x.unsqueeze(1))  # Pass through RNN layer
        x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function and optimizer
model = CNN_RNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print("Model, loss function, and optimizer initialized")

# Train the model
for epoch in range(10):
    print(f"Starting epoch {epoch + 1}")
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Print accuracy after each epoch
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Validation accuracy after epoch %d: %d %%' % (epoch + 1, 100 * correct / total))

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Test accuracy: %d %%' % (100 * correct / total))

# Save the model
torch.save(model.state_dict(), 'CNN_RNNv1.pth')
print("Model saved successfully")