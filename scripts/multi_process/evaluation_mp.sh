#!/bin/bash

# =============================================================================
# Multi-Processing Evaluation Script
# =============================================================================
# This script runs evaluator.py with parallel processing:
# - Datasets are processed sequentially (to avoid conflicts)
# - Within each dataset, different chunk_sizes run in parallel
# 
# Usage:
#   source ./parameters.sh
#   ./mp/evaluation_mp.sh
# =============================================================================

# Default parameters (can be overridden by environment variables)
predict_data_dir=${predict_data_dir:-"./datahub_outputs/3_predict"}
metric_data_dir=${metric_data_dir:-"./datahub_outputs/4_metric"}
split_seed=${split_seed:-42}
test_chunk_size=${test_chunk_size:-50}
model_name=${model_name:-"minzl/toy3_2800"}
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
    echo "Step 2: Run parallel batch evaluation"
    echo "  ./mp/evaluation_mp.sh"
    echo ""
    echo "Alternatively, you can set environment variables manually:"
    echo "  export dataset_names=\"bank heloc led7\""
    echo "  export train_chunk_sizes=\"8 32 128\""
    echo "  export row_shuffle_seeds=\"40 41 42 43 44\""
    echo "  export model_name=\"minzl/toy3_2800\""
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
echo "Multi-Processing Batch Evaluation Configuration"
echo "=============================================================================="
echo "Predict data directory: $predict_data_dir"
echo "Metric data directory: $metric_data_dir"
echo "Datasets: ${DATASETS[*]}"
echo "Train chunk sizes: ${CHUNK_SIZES[*]}"
echo "Row shuffle seeds: ${SHUFFLE_SEEDS[*]}"
echo "Model name: $model_name"
echo "Split seed: $split_seed"
echo "Test chunk size: $test_chunk_size"
echo "Max parallel jobs: $max_parallel_jobs"
echo "Wait between datasets: ${wait_between_datasets}s"
echo "=============================================================================="
echo ""

# Function to run a single evaluation configuration
run_single_evaluation() {
    local dataset=$1
    local train_chunk_size=$2
    local job_id=$3
    
    echo "STARTING: [Job $job_id] Starting evaluation: $dataset (chunk_size: $train_chunk_size)"
    
    # Run the Python script
    python ./evaluation/result_proc/evaluator.py \
        --input_dir "$predict_data_dir" \
        --output_dir "$metric_data_dir" \
        --dataset_name "$dataset" \
        --model_name "$model_name" \
        --split_seed "$split_seed" \
        --row_shuffle_seeds ${SHUFFLE_SEEDS[*]} \
        --train_chunk_size "$train_chunk_size" \
        --test_chunk_size "$test_chunk_size"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS: [Job $job_id] Completed evaluation: $dataset (chunk_size: $train_chunk_size)"
    else
        echo "ERROR: [Job $job_id] Failed evaluation: $dataset (chunk_size: $train_chunk_size) [Exit code: $exit_code]"
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
total_chunk_sizes_per_dataset=${#CHUNK_SIZES[@]}
total_evaluations=$((${#DATASETS[@]} * total_chunk_sizes_per_dataset))
overall_job_counter=0

echo "📊 Processing Summary:"
echo "   - ${#DATASETS[@]} datasets"
echo "   - ${#CHUNK_SIZES[@]} chunk sizes per dataset"
echo "   - Total evaluations: $total_evaluations"
echo "   - Row shuffle seeds per evaluation: ${SHUFFLE_SEEDS[*]}"
echo "   - Max parallel jobs: $max_parallel_jobs"
echo ""

for dataset_idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$dataset_idx]}"
    
    echo "================================================================================"
    echo "Input: Processing Dataset: $dataset ($(($dataset_idx + 1))/${#DATASETS[@]})"
    echo "================================================================================"
    
    # Launch parallel jobs for all chunk_sizes for this dataset
    for train_chunk_size in "${CHUNK_SIZES[@]}"; do
        overall_job_counter=$((overall_job_counter + 1))
        
        # Wait if we've reached the maximum number of parallel jobs
        wait_for_jobs $max_parallel_jobs
        
        # Launch job in background
        run_single_evaluation "$dataset" "$train_chunk_size" "$overall_job_counter" &
        
        # Show progress
        printf "⚡ Launched [Job %2d/%2d]: %s (chunk_size: %s)\n" \
            "$overall_job_counter" "$total_evaluations" "$dataset" "$train_chunk_size"
    done
    
    # Wait for all jobs for this dataset to complete before moving to next dataset
    echo ""
    echo "⏳ Waiting for all evaluation jobs for dataset '$dataset' to complete..."
    wait
    
    echo "SUCCESS: All evaluation jobs for dataset '$dataset' completed!"
    
    # Wait between datasets (except for the last one)
    if [ $dataset_idx -lt $((${#DATASETS[@]} - 1)) ]; then
        echo "😴 Waiting ${wait_between_datasets}s before processing next dataset..."
        sleep $wait_between_datasets
        echo ""
    fi
done

echo ""
echo "Completed: Multi-processing batch evaluation completed!"
echo "📊 Processing Summary:"
echo "   - Processed ${#DATASETS[@]} datasets"
echo "   - Processed ${#CHUNK_SIZES[@]} chunk sizes per dataset"
echo "   - Total evaluations: $total_evaluations"
echo "   - Row shuffle seeds used: ${SHUFFLE_SEEDS[*]}"
echo "⚡ Used up to $max_parallel_jobs parallel jobs per dataset"
echo "Output: Results saved to: $metric_data_dir"
