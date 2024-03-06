import os
import csv
from collections import defaultdict

# Define the root directory for all datasets we have
root_dir = "/raid/zihan/dataset"

# Create a dictionary to keep track of the number of files processed per model
model_file_count = defaultdict(lambda: [0, 0, 0])  # [train, validate, evaluate]

# Open the output files in write mode
with open('train.csv', 'w', newline='') as f_train, \
     open('validate.csv', 'w', newline='') as f_validate, \
     open('evaluate.csv', 'w', newline='') as f_evaluate:

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

                # Get the model name from the name
                model_name = name.split('*')[0]

                # Determine the label
                label = 'bonafide' if model_name == 'IDMA' else 'spoof'

                # Write the name, full path, and label to the appropriate .csv file
                if model_name == 'IDMA':
                    if model_file_count[model_name][0] < 5500:  # Training set
                        writer_train.writerow([name, full_path, label])
                        model_file_count[model_name][0] += 1
                    elif model_file_count[model_name][1] < 1500:  # Validation set
                        writer_validate.writerow([name, full_path, label])
                        model_file_count[model_name][1] += 1
                    elif model_file_count[model_name][2] < 1300:  # Evaluation set
                        writer_evaluate.writerow([name, full_path, label])
                        model_file_count[model_name][2] += 1
                else:
                    if model_file_count[model_name][0] < 700:  # Training set
                        writer_train.writerow([name, full_path, label])
                        model_file_count[model_name][0] += 1
                    elif model_file_count[model_name][1] < 300:  # Validation set
                        writer_validate.writerow([name, full_path, label])
                        model_file_count[model_name][1] += 1
                    elif model_file_count[model_name][2] < 300:  # Evaluation set
                        writer_evaluate.writerow([name, full_path, label])
                        model_file_count[model_name][2] += 1