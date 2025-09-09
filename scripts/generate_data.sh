#!/bin/bash
# Description: This script generates SCM data and converts it to CSV format
# Note: Must be executed in an environment with the tabicl library available

# ===================== Configuration Parameters =====================
# The 'id' variable serves as a unique identifier.
# By splitting data generation across multiple batches (using different id values),
# we can parallelize the process and significantly accelerate overall data generation.
export id=1                  # Dataset identifier 

export device="cpu"          # Computation device (cpu/cuda)
export np_seed=$((123+id*100))      # NumPy random seed (depends on id)
export torch_seed=$((123+id*100))   # PyTorch random seed (depends on id)

# Path configurations
export ptfolder="./prior_data/scmptfile_$id"    # PyTorch data storage directory
# The final generated CSV data will be stored in this directory:
export csvfolder="./prior_data/scmcsvfile_$id"  # CSV data storage directory

# ===================== Output Location Information =====================
echo "=================================================================="
echo "CSV output will be saved to: $csvfolder"
echo "=================================================================="

# ===================== Display Seed Information =====================
echo "Processing Data ID: $id"
echo "NumPy Seed: $np_seed"
echo "Torch Seed: $torch_seed"

# ===================== Data Generation Phase =====================
echo "Starting data generation for ID $id..."
python ./third_party/tabicl/src/tabicl/prior/genload.py \
    --save_dir        "$ptfolder" \
    --device          "$device" \
    --num_batches      100 \
    --batch_size       1 \
    --batch_size_per_gp 1 \
    --min_features     5 \
    --max_features     50 \
    --max_classes      10 \
    --min_seq_len      1024 \
    --max_seq_len      1025 \
    --prior_type       "mix_scm" \
    --n_jobs           4 \
    --num_threads_per_generate 1 \
    --np_seed          "$np_seed" \
    --torch_seed       "$torch_seed"

# ===================== Format Conversion Phase =====================
echo "Converting ID $id data to CSV format..."
python ./src/prior_data/pt_to_csv.py "$ptfolder" "$csvfolder" 

echo "ID $id processing completed successfully!"
echo "=================================================================="
echo "Final CSV data available at: $csvfolder"
echo "=================================================================="