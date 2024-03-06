import os
import csv
from collections import defaultdict

# Define the root directory for all datasets we have
root_dir = "/path/to/root/dataset"

# Create a dictionary to keep track of the number of files processed per label
label_file_count = defaultdict(lambda: [0, 0, 0])  # [train, validate, evaluate]

# Open the output files in write mode
with open('../csvFilesReduced/train.csv', 'w', newline='') as f_train, \
     open('../csvFilesReduced/validate.csv', 'w', newline='') as f_validate, \
     open('../csvFilesReduced/evaluate.csv', 'w', newline='') as f_evaluate:

    writer_train = csv.writer(f_train, delimiter=',')
    writer_validate = csv.writer(f_validate, delimiter=',')
    writer_evaluate = csv.writer(f_evaluate, delimiter=',')

    # Write the headers
    writer_train.writerow(['name', 'full_path', 'label'])
    writer_validate.writerow(['name', 'full_path', 'label'])
    writer_evaluate.writerow(['name', 'full_path', 'label'])

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
