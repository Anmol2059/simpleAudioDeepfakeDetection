import streamlit as st
import torch
import numpy as np
import librosa
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import librosa.display

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

# Function to preprocess the audio files and extract spectrograms
def extract_features(file):
    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print(f"Error encountered while parsing file. Error details: {e}")
        return None, None, None
     
    return mfccsscaled, sample_rate, mfccs

# Load the model
model = CNN_RNN()
model.load_state_dict(torch.load('CNN_RNNv2.pth'))
model.eval()

# Streamlit app
st.title('Audio Deepfake Detection App')

# Record audio feature
uploaded_file = st.file_uploader("Or upload a .wav file", type="wav")
if not uploaded_file:
    st.warning('Please upload a .wav file or record audio using the button below.')

if uploaded_file is not None:
    audio_data, sample_rate, mfccs = extract_features(uploaded_file)
    if audio_data is not None:
        st.audio(uploaded_file, format='audio/wav', start_time=0)
        
        audio_data_tensor = torch.from_numpy(audio_data.reshape(-1, 1, audio_data.shape[0], 1))
        with torch.no_grad():
            output = model(audio_data_tensor.float())
            proba = F.softmax(output, dim=1)
            confidence_score, predicted = torch.max(proba, 1)
        
        if predicted.item() == 0:
            st.write(f'Predicted class: bonafide with confidence score: {confidence_score.item()}', unsafe_allow_html=True)
            st.write('<span style="color:green;">bonafide</span>', unsafe_allow_html=True)
        else:
            st.write(f'Predicted class: spoof with confidence score: {confidence_score.item()}', unsafe_allow_html=True)
            st.write('<span style="color:red;">spoof</span>', unsafe_allow_html=True)

        # Display audio waveform
        st.subheader('Audio Waveform')
        plt.figure(figsize=(10, 2))
        plt.plot(np.linspace(0, len(audio_data)/sample_rate, len(audio_data)), audio_data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Audio Waveform')
        st.pyplot(plt)

        # Display spectrogram
        st.subheader('Spectrogram')
        plt.figure(figsize=(10, 4))
        plt.specgram(audio_data, Fs=sample_rate)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram')
        st.pyplot(plt)

        # Display MFCCs
        st.subheader('MFCCs')
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCCs')
        st.pyplot(plt)

        # Get the probabilities for the 'bonafide' and 'spoof' classes
        proba_np = proba.numpy()[0][:2]

        # Display class probabilities
        st.subheader('Class Probabilities')
        classes = ['bonafide', 'spoof']
        plt.bar(classes, proba_np)
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        st.pyplot(plt)

        # Explain the class probabilities in plain language
        bonafide_prob = proba_np[0]
        spoof_prob = proba_np[1]
        st.write(f"The model is {bonafide_prob * 100:.2f}% confident that the audio is genuine (bonafide), and {spoof_prob * 100:.2f}% confident that the audio is fake (spoof).")

        # Display model architecture
        st.text(str(model))

    else:
        st.error('Could not extract features from the audio file.')