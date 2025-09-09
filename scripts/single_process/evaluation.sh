#!/bin/bash

# =============================================================================
# Evaluation Script
# =============================================================================
# This script runs evaluator.py for multiple datasets and chunk sizes
# 
# Usage:
#   source ./parameters.sh
#   ./scripts/single_process/evaluation.sh
# =============================================================================

# Default parameters (can be overridden by environment variables)
predict_data_dir=${predict_data_dir:-"./datahub_outputs/3_predict"}
metric_data_dir=${metric_data_dir:-"./data/outputs/4_metric"}
split_seed=${split_seed:-42}
test_chunk_size=${test_chunk_size:-50}
model_name=${model_name:-"minzl/toy3_2800"}

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
    echo "Step 2: Run batch evaluation"
    echo "  ./scripts/single_process/evaluation.sh"
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
echo "Batch Evaluation Configuration"
echo "=============================================================================="
echo "Predict data directory: $predict_data_dir"
echo "Metric data directory: $metric_data_dir"
echo "Datasets: ${DATASETS[*]}"
echo "Train chunk sizes: ${CHUNK_SIZES[*]}"
echo "Row shuffle seeds: ${SHUFFLE_SEEDS[*]}"
echo "Model name: $model_name"
echo "Split seed: $split_seed"
echo "Test chunk size: $test_chunk_size"
echo "=============================================================================="
echo ""

# Counter for tracking progress
total_jobs=$((${#DATASETS[@]} * ${#CHUNK_SIZES[@]}))
current_job=0

# Main processing loop (2-level nested: dataset -> chunk_size)
for dataset in "${DATASETS[@]}"; do
    for train_chunk_size in "${CHUNK_SIZES[@]}"; do
        current_job=$((current_job + 1))
        
        # Create elegant table-style output
        printf "┌─────────────────────────────────────────────────────────────────────────────────────────┐\n"
        printf "│ Job %2d/%-2d │ Dataset: %-15s │ Chunk Size: %-8s │\n" "$current_job" "$total_jobs" "$dataset" "$train_chunk_size"
        printf "└─────────────────────────────────────────────────────────────────────────────────────────┘\n"
        
        # Run the actual command
        python ./src/evaluation/result_proc/evaluator.py \
            --input_dir "$predict_data_dir" \
            --output_dir "$metric_data_dir" \
            --dataset_name "$dataset" \
            --model_name "$model_name" \
            --split_seed "$split_seed" \
            --row_shuffle_seeds ${SHUFFLE_SEEDS[*]} \
            --train_chunk_size "$train_chunk_size" \
            --test_chunk_size "$test_chunk_size"
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "✅ Successfully evaluated $dataset (chunk_size: $train_chunk_size)"
        else
            echo "❌ Failed to evaluate $dataset (chunk_size: $train_chunk_size)"
            echo "Error code: $?"
        fi
        echo "==================================================================================="
        echo ""
    done
done

echo "🎉 Batch evaluation completed!"
echo "📊 Processing Summary:"
echo "   - Processed ${#DATASETS[@]} datasets"
echo "   - Processed ${#CHUNK_SIZES[@]} chunk sizes"
echo "   - Total evaluation jobs: $total_jobs"
echo "   - Row shuffle seeds used: ${SHUFFLE_SEEDS[*]}"
echo "📁 Results saved to: $metric_data_dir"
