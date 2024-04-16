import os
import csv

# Define the path to the image directory
directory_path = 'dataset/dogs_cats/train'

# CSV file to store the image names and labels
output_csv_path = 'dataset/dogs_cats/labels.csv'

# Open the CSV file for writing
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['filename', 'label'])

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):  # Check if the file is a JPEG image
            # Check if the filename starts with 'cat' or 'dog' and assign the label
            if filename.startswith('cat'):
                label = 0
            elif filename.startswith('dog'):
                label = 1
            else:
                continue  # Skip files that do not match the expected naming convention

            # Write the filename and label to the CSV
            writer.writerow([filename, label])

print("CSV file has been created at:", output_csv_path)
