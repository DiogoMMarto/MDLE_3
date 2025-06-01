import csv
import argparse

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Convert a .dat file to a .csv file.')
parser.add_argument('input_file', type=str, help='Path to the input .dat file')
parser.add_argument('output_file', type=str, help='Path to the output .csv file')
args = parser.parse_args()

# Get input and output file paths from command line arguments
input_file = args.input_file
output_file = args.output_file

# Open the input .dat file and output .csv file
with open(input_file, 'r') as dat_file, open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write CSV header
    writer.writerow(['userID', 'movieId', 'rating', 'timestamp'])

    # Process each line in the .dat file
    for line in dat_file:
        # Split the line by "::"
        parts = line.strip().split("::")
        if len(parts) == 4:
            writer.writerow(parts)

print(f"Conversion complete. CSV saved as '{output_file}'.")
