import os
import csv
import shutil
from collections import defaultdict

# Define the root directory for all datasets we have
root_dir = os.path.dirname(os.path.realpath(__file__))

# Create a dictionary to keep track of the number of files processed per label
label_file_count = defaultdict(lambda: [0, 0, 0])  # [train, validate, evaluate]

# If the csvFiles directory exists, remove it and create a new one
csv_dir = os.path.join(root_dir, 'csvFiles')
if os.path.exists(csv_dir):
    shutil.rmtree(csv_dir)
os.makedirs(csv_dir)

# Open the output files in write mode
with open(os.path.join(csv_dir, 'train.csv'), 'w', newline='') as f_train, \
     open(os.path.join(csv_dir, 'validate.csv'), 'w', newline='') as f_validate, \
     open(os.path.join(csv_dir, 'evaluate.csv'), 'w', newline='') as f_evaluate:

    writer_train = csv.writer(f_train, delimiter=',')
    writer_validate = csv.writer(f_validate, delimiter=',')
    writer_evaluate = csv.writer(f_evaluate, delimiter=',')

    # Walk through the directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file ends with .wav or .WAV, can add other formats if available
            if filename.endswith(('.wav', '.WAV')):
                # Construct the full path
                full_path = os.path.join(dirpath, filename)

                # Construct the name by replacing the root directory and .wav or .WAV extension
                name = full_path.replace(root_dir + '/', '').replace('.wav', '').replace('.WAV', '').replace('/', '*')

                # Determine the label
                label = 'bonafide' if 'real' in dirpath else 'spoof'

                # Split the data into a 7:2:1 ratio for training, validation, and evaluation sets
                split = sum(label_file_count[label]) % 10
                if split < 7:  # Training set
                    writer_train.writerow([name, full_path, label])
                    label_file_count[label][0] += 1
                elif split < 9:  # Validation set
                    writer_validate.writerow([name, full_path, label])
                    label_file_count[label][1] += 1
                else:  # Evaluation set
                    writer_evaluate.writerow([name, full_path, label])
                    label_file_count[label][2] += 1