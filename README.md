# Audio Classification Project

This project is about classifying audio files using a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). The project consists of two main Python scripts: `train1.py` and `app.py`.

## train1.py

This script is responsible for training the model. It first loads the data from CSV files, then preprocesses the audio files to extract MFCC features. The preprocessed data is saved to disk for future use. The script then defines a CNN-RNN model, trains it on the training data, and evaluates it on the validation and test data. The trained model is saved to disk.

## app.py

This script is responsible for the application that uses the trained model to classify audio files. It uses the Streamlit library to create a web-based user interface.

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```
## Usage

**Training:**

```bash
python train1.py
```
**Running Application**
```bash
python app.py
```