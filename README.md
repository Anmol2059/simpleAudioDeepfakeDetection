

# Deepfake Speech Detection Project

This project focuses on detecting deepfake speech using a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN).

## Project Outline

1. **Data Preparation**
   - Audio files are organized into a directory structure with 'data/fake' and 'data/real' subdirectories.
   - The `generateDataCSV.py` script is used to generate CSV files for organizing the audio dataset into training, validation, and evaluation sets.

2. **Data Preprocessing**
   - The `train1.py` script preprocesses the audio files to extract MFCC features.
   - MFCC features are saved to disk for future use.

3. **Model Training**
   - The `train1.py` script defines a CNN-RNN model and trains it on the preprocessed data.
   - The trained model is evaluated on the validation and test data.

4. **Model Evaluation**
   - The `eval.py` script evaluates the trained model on the test data.

5. **Running the Application**
   - The `app.py` script uses the trained model to classify audio files and creates a web-based user interface using Streamlit.

## Step-by-Step Guide

### 1. Prepare Data

1. Place your audio dataset in the following directory structure:
   ```
   /path/to/root/dataset/
   ├── data
   │   ├── fake
   │   └── real
   ├── generateDataCSV.py
   ├── train1.py
   ├── eval.py
   └── app.py
   ```

2. Run the `generateDataCSV.py` script to generate CSV files for organizing the audio dataset:
   ```bash
   python generateDataCSV.py
   ```
   - The generated CSV files will be saved in the `csvFilesReduced` directory, as evaluate train and validate.csv.

### 2. Train the Model

1. Run the `train.py` script to train the model:
   ```bash
   python train.py
   ```
1.1. You may also Run the `trainProcessedSample.py` script to train the model, here features are already extracted of a large dataset:
   ```bash
   python trainProcessedSample.py
   ```

### 3. Evaluate the Model

1. Run the `eval.py` script to evaluate the trained model:
   ```bash
   python eval.py
   ```

### 4. Run the Application

1. Run the `app.py` script to start the application:
   ```bash
   streamlit run app.py
   ```

## Notes

- This is a basic model implementation. Feel free to modify and enhance it based on your requirements and dataset.

---
