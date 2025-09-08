import os
import torch
import pandas as pd
from tqdm import tqdm
import argparse

def process_pt_file(file_path, output_dir):
    # Load the .pt file
    data = torch.load(file_path)
    X = data['X']
    y = data['y']
    seq_lens = data['seq_lens']
    d = data['d']
    batch_size = data['batch_size']

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each batch
    start_idx = 0
    for i in range(batch_size):
        # Determine seq_len and d for this batch
        seq_len = seq_lens[i].item()
        feature_dim = d[i].item()

        # Extract features and labels for this batch
        end_idx = start_idx + seq_len * feature_dim
        batch_features = X[start_idx:end_idx].reshape(seq_len, feature_dim)
        batch_labels = y[i]

        # Create DataFrame for this batch
        df = pd.DataFrame(batch_features.numpy())
        df = ((df * 120 + 500).clip(lower=0)).astype(int)
        df['label'] = batch_labels.numpy()
        df['label'] = df['label'].astype(int)

        # Save to CSV
        base_filename = os.path.basename(file_path)
        filename = f"{base_filename[:-3]}_{i}.csv"
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)

        # Update start index for the next batch
        start_idx = end_idx

def process_pt_files(input_dir, output_dir):
    # Iterate over all .pt files in the input directory
    pt_files = [filename for filename in os.listdir(input_dir) if filename.endswith('.pt')]

    # Create a progress bar for processing .pt files
    for filename in tqdm(pt_files, desc="Processing .pt files"):
        file_path = os.path.join(input_dir, filename)
        process_pt_file(file_path, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Convert .pt files to CSV format.')
    parser.add_argument('input_directory', type=str, help='Directory containing .pt files')
    parser.add_argument('output_directory', type=str, help='Directory to save CSV files')
    args = parser.parse_args()

    # Process files
    process_pt_files(args.input_directory, args.output_directory)

if __name__ == "__main__":
    main()
