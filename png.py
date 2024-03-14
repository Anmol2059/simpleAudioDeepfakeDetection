from torchviz import make_dot
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
# Assuming your model is named `model`
model = CNN_RNN().to(device)

# Create a random tensor that matches the input size your model expects.
# For example, if your model expects input of size (1, 1, 20, 1), you can create a tensor like this:
x = torch.randn(32, 1, 40, 1).to(device)

# Run the tensor through the model so that all the operations are recorded.
y = model(x)

# Generate a graph of the model
dot = make_dot(y, params=dict(model.named_parameters()))

# Save the graph to a file
dot.format = 'png'
dot.render('model2_architecture')