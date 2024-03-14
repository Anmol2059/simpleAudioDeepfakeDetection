import pandas as pd
import numpy as np
import librosa
import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from torch import flatten
from torch.nn import functional as F
import pickle

import os
if torch.cuda.is_available() and torch.cuda.device_count() > 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device:", device)

# Load the data
train_data = pd.read_csv('csvFiles/train.csv', names=['name', 'path', 'label'])
validation_data = pd.read_csv('csvFiles/validate.csv', names=['name', 'path', 'label'])
evaluation_data = pd.read_csv('csvFiles/evaluate.csv', names=['name', 'path', 'label'])

print("Datasets loaded successfully")

# Function to preprocess the audio files and extract spectrograms
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}. Error details: {e}")
        return None 
     
    return mfccsscaled

# Function to process the data

def process_data(df, X_save_path, y_save_path):

    features = []
    total_files = len(df)
    for i, (index, row) in enumerate(df.iterrows(), 1):
        file_name = row['path']
        class_label = row['label']
        data = extract_features(file_name)
        features.append([data, class_label])
        print(f"Processed {i}/{total_files} files", end='\r')  # \r to overwrite the previous print

    # Convert into a Panda dataframe 
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    os.makedirs(os.path.dirname(X_save_path), exist_ok=True)
    # Save the processed data
    np.save(X_save_path, X)
    np.save(y_save_path, y)
    print(f"Saved processed data to {X_save_path} and {y_save_path}")

    print("\nData processing done")
    return X, y

# Process the data
x_train, y_train = process_data(train_data, 'processedData/train_data_X.npy', 'processedData/train_data_y.npy')
x_val, y_val = process_data(validation_data, 'processedData/val_data_X.npy', 'processedData/val_data_y.npy')
x_test, y_test = process_data(evaluation_data, 'processedData/test_data_X.npy', 'processedData/test_data_y.npy')
print("Data processing completed")

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
        self.rnn = nn.RNN(32*20*1, 64, batch_first=True)  # RNN layer
        self.fc1 = nn.Linear(64, 128)  # Adjusted input features
        self.fc2 = nn.Linear(128, 2)  # Adjusted output features

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
# model = torch.load('CNN_RNNv1.pth', map_location=device)
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

    # Print accuracy after each epoch validation set
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
all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Store the true labels and the predicted labels
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

print('Test accuracy: %d %%' % (100 * correct / total))

# Calculate F1 score, precision, and recall
f1 = f1_score(all_labels, all_predictions, average='weighted')
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')

print(f'F1 Score: {f1:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# Save the model
torch.save(model.state_dict(), 'CNN_RNNv2.pth')
print("Model saved successfully")