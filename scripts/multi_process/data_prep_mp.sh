#!/bin/bash

# =============================================================================
# Multi-Processing Data Preparation Script
# =============================================================================
# This script runs data_chunk_prep.py with parallel processing:
# - Datasets are processed sequentially (to avoid conflicts)
# - Within each dataset, (chunk_size, seed) combinations run in parallel
# 
# Usage:
#   source ./parameters.sh
#   ./scripts/multi_process/data_prep_mp.sh
# =============================================================================

# Default parameters (can be overridden by environment variables)
original_data_dir=${original_data_dir:-"./datahub_inputs/data_raw"}
split_data_dir=${split_data_dir:-"./datahub_outputs/1_split"}
split_seed=${split_seed:-42}
test_chunk_size=${test_chunk_size:-50}
test_size=${test_size:-0.2}
shuffle_columns=${shuffle_columns:-True}
max_workers=${max_workers:-2}
force_overwrite=${force_overwrite:-True}
max_parallel_jobs=${max_parallel_jobs:-6}
wait_between_datasets=${wait_between_datasets:-5}

# Required parameters (must be set via environment variables)
dataset_names=${dataset_names:-""}
train_chunk_sizes=${train_chunk_sizes:-""}
row_shuffle_seeds=${row_shuffle_seeds:-""}

# Function to print usage
print_usage() {
    echo "Usage: This script requires parameters to be loaded from parameters.sh"
    echo ""
    echo "Step 1: Load parameters"
    echo "  source ./parameters.sh"
    echo ""
    echo "Step 2: Run parallel batch processing"
    echo "  ./scripts/multi_process/batch_data_prep_mp.sh"
    echo ""
    echo "Alternatively, you can set environment variables manually:"
    echo "  export dataset_names=\"bank heloc rl\""
    echo "  export train_chunk_sizes=\"8 32 128\""
    echo "  export row_shuffle_seeds=\"40 41 42\""
    echo "  # ... other parameters"
}

# Check if required parameters are set
if [ -z "$dataset_names" ] || [ -z "$train_chunk_sizes" ] || [ -z "$row_shuffle_seeds" ]; then
    echo "Error: Required parameters not set!"
    echo ""
    print_usage
    exit 1
fi

# Convert space-separated strings to arrays
read -ra DATASETS <<< "$dataset_names"
read -ra CHUNK_SIZES <<< "$train_chunk_sizes"
read -ra SHUFFLE_SEEDS <<< "$row_shuffle_seeds"

# Print configuration
echo "=============================================================================="
echo "Multi-Processing Batch Data Preparation Configuration"
echo "=============================================================================="
echo "Original data directory: $original_data_dir"
echo "Split data directory: $split_data_dir"
echo "Datasets: ${DATASETS[*]}"
echo "Train chunk sizes: ${CHUNK_SIZES[*]}"
echo "Row shuffle seeds: ${SHUFFLE_SEEDS[*]}"
echo "Split seed: $split_seed"
echo "Test chunk size: $test_chunk_size"
echo "Test size: $test_size"
echo "Shuffle columns: $shuffle_columns"
echo "Max workers per job: $max_workers"
echo "Max parallel jobs: $max_parallel_jobs"
echo "Wait between datasets: ${wait_between_datasets}s"
echo "Force overwrite: $force_overwrite"
echo "=============================================================================="
echo ""

# Function to run a single configuration
run_single_config() {
    local dataset=$1
    local train_chunk_size=$2
    local row_shuffle_seed=$3
    local job_id=$4
    
    echo "🚀 [Job $job_id] Starting: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
    
    # Run the Python script
    python ./src/evaluation/data_prep/data_chunk_prep.py \
        --input_dir "$original_data_dir" \
        --output_dir "$split_data_dir" \
        --dataset_name "$dataset" \
        --split_seed "$split_seed" \
        --row_shuffle_seed "$row_shuffle_seed" \
        --train_chunk_size "$train_chunk_size" \
        --test_chunk_size "$test_chunk_size" \
        --test_size "$test_size" \
        --shuffle_columns "$shuffle_columns" \
        --max_workers "$max_workers" \
        --force_overwrite
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ [Job $job_id] Completed: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
    else
        echo "❌ [Job $job_id] Failed: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed) [Exit code: $exit_code]"
    fi
    
    return $exit_code
}

# Function to wait for jobs to complete
wait_for_jobs() {
    local max_jobs=$1
    
    # Wait until number of background jobs is less than max_jobs
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 1
    done
}

# Main processing loop
total_configs_per_dataset=$((${#CHUNK_SIZES[@]} * ${#SHUFFLE_SEEDS[@]}))
total_configs=$((${#DATASETS[@]} * total_configs_per_dataset))
overall_job_counter=0

echo "📊 Processing Summary:"
echo "   - ${#DATASETS[@]} datasets"
echo "   - ${#CHUNK_SIZES[@]} chunk sizes × ${#SHUFFLE_SEEDS[@]} shuffle seeds = $total_configs_per_dataset configs per dataset"
echo "   - Total configurations: $total_configs"
echo "   - Max parallel jobs: $max_parallel_jobs"
echo ""

for dataset_idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$dataset_idx]}"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📂 Processing Dataset: $dataset ($(($dataset_idx + 1))/${#DATASETS[@]})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Launch parallel jobs for all (chunk_size, seed) combinations for this dataset
    for train_chunk_size in "${CHUNK_SIZES[@]}"; do
        for row_shuffle_seed in "${SHUFFLE_SEEDS[@]}"; do
            overall_job_counter=$((overall_job_counter + 1))
            
            # Wait if we've reached the maximum number of parallel jobs
            wait_for_jobs $max_parallel_jobs
            
            # Launch job in background
            run_single_config "$dataset" "$train_chunk_size" "$row_shuffle_seed" "$overall_job_counter" &
            
            # Show progress
            printf "⚡ Launched [Job %2d/%2d]: %s (chunk:%s, seed:%s)\n" \
                "$overall_job_counter" "$total_configs" "$dataset" "$train_chunk_size" "$row_shuffle_seed"
        done
    done
    
    # Wait for all jobs for this dataset to complete before moving to next dataset
    echo ""
    echo "⏳ Waiting for all jobs for dataset '$dataset' to complete..."
    wait
    
    echo "✅ All jobs for dataset '$dataset' completed!"
    
    # Wait between datasets (except for the last one)
    if [ $dataset_idx -lt $((${#DATASETS[@]} - 1)) ]; then
        echo "😴 Waiting ${wait_between_datasets}s before processing next dataset..."
        sleep $wait_between_datasets
        echo ""
    fi
done

echo ""
echo "🎉 Multi-processing batch processing completed!"
echo "📊 Processed $total_configs total configurations across ${#DATASETS[@]} datasets"
echo "⚡ Used up to $max_parallel_jobs parallel jobs per dataset"
