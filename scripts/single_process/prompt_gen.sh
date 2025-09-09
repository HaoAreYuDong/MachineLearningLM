#!/bin/bash

# =============================================================================
# Prompt Generation Script
# =============================================================================
# This script runs data_prompt_gen.py for multiple datasets, chunk sizes, and seeds
# 
# Usage:
#   source ./parameters.sh
#   ./scripts/single_process/prompt_gen.sh
# =============================================================================

# Default parameters (can be overridden by environment variables)
split_data_dir=${split_data_dir:-"./datahub_outputs/1_split"}
prompt_data_dir=${prompt_data_dir:-"./datahub_outputs/2_prompt"}
split_seed=${split_seed:-42}
test_chunk_size=${test_chunk_size:-50}
max_workers=${max_workers:-8}
force_overwrite=${force_overwrite:-True}
normalization=${normalization:-True}
include_feature_descriptions=${include_feature_descriptions:-False}
prompt_format_style=${prompt_format_style:-"concat"}

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
    echo "Step 2: Run batch prompt generation"
    echo "  ./scripts/single_process/prompt_gen.sh"
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
echo "Batch Prompt Generation Configuration"
echo "=============================================================================="
echo "Split data directory: $split_data_dir"
echo "Prompt data directory: $prompt_data_dir"
echo "Datasets: ${DATASETS[*]}"
echo "Train chunk sizes: ${CHUNK_SIZES[*]}"
echo "Row shuffle seeds: ${SHUFFLE_SEEDS[*]}"
echo "Split seed: $split_seed"
echo "Test chunk size: $test_chunk_size"
echo "Normalization: $normalization"
echo "Include feature descriptions: $include_feature_descriptions"
echo "Prompt format style: $prompt_format_style"
echo "Max workers: $max_workers"
echo "Force overwrite: $force_overwrite"
echo "=============================================================================="
echo ""

# Counter for tracking progress
total_jobs=$((${#DATASETS[@]} * ${#CHUNK_SIZES[@]} * ${#SHUFFLE_SEEDS[@]}))
current_job=0

# Main processing loop (3-level nested: dataset -> chunk_size -> row_shuffle_seed)
for dataset in "${DATASETS[@]}"; do
    for train_chunk_size in "${CHUNK_SIZES[@]}"; do
        for row_shuffle_seed in "${SHUFFLE_SEEDS[@]}"; do
            current_job=$((current_job + 1))
            
            # Create elegant table-style output
            printf "┌─────────────────────────────────────────────────────────────────────────────────────────┐\n"
            printf "│ Job %2d/%-2d │ Dataset: %-10s │ Chunk: %-4s │ Shuffle Seed: %-4s │\n" "$current_job" "$total_jobs" "$dataset" "$train_chunk_size" "$row_shuffle_seed"
            printf "└─────────────────────────────────────────────────────────────────────────────────────────┘\n"
            
            # Run the actual command
            python ./src/evaluation/prompt_gen/data_prompt_gen.py \
                --input_dir "$split_data_dir" \
                --output_dir "$prompt_data_dir" \
                --dataset_name "$dataset" \
                --split_seed "$split_seed" \
                --row_shuffle_seed "$row_shuffle_seed" \
                --train_chunk_size "$train_chunk_size" \
                --test_chunk_size "$test_chunk_size" \
                --normalization "$normalization" \
                --include_feature_descriptions "$include_feature_descriptions" \
                --prompt_format_style "$prompt_format_style" \
                --max_workers "$max_workers" \
                --force_overwrite
            
            # Check if the command was successful
            if [ $? -eq 0 ]; then
                echo "✅ Successfully processed $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
            else
                echo "❌ Failed to process $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
                echo "Error code: $?"
            fi
            echo "==================================================================================="
            echo ""
        done
    done
done

echo "🎉 Batch prompt generation completed!"
echo "Processed $total_jobs jobs for ${#DATASETS[@]} datasets, ${#CHUNK_SIZES[@]} chunk sizes, and ${#SHUFFLE_SEEDS[@]} shuffle seeds."
