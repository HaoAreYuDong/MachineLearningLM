#!/bin/bash

# =============================================================================
# Optimized Batch Model Prediction Script (ML & DL Support)
# =============================================================================
# This script automatically selects between ML and DL model prediction based on model_name:
# - ML models (randomforest, rf, knn, xgboost, xgb): uses ml_model_pred.py with 1_split data
# - DL models (all others): uses dl_model_pred.py with 2_prompt data
# - Model is loaded once per chunk_size (instead of once per task)
# - Processes all datasets and seeds for each chunk_size in one go
# 
# Usage:
#   source ./parameters.sh
#   ./scripts/multi_process/model_pred.sh
# =============================================================================

# Default parameters (can be overridden by environment variables)
prompt_data_dir=${prompt_data_dir:-"./datahub_outputs/2_prompt"}
predict_data_dir=${predict_data_dir:-"./datahub_outputs/3_predict"}
split_seed=${split_seed:-42}
test_chunk_size=${test_chunk_size:-50}
force_overwrite=${force_overwrite:-True}
model_name=${model_name:-"openai::gpt-4o-mini"}
temperature=${temperature:-0.0}
max_samples=${max_samples:-""}
max_workers=${max_workers:-""}
labels=${labels:-""}
device_id=${device_id:-"0"}
logprobs_supported=${logprobs_supported:-"True"}

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
    echo "Step 2: Run batch model prediction"
    echo "  ./scripts/multi_process/model_pred.sh"
    echo ""
    echo "Alternatively, you can set environment variables manually:"
    echo "  export dataset_names=\"bank heloc rl\""
    echo "  export train_chunk_sizes=\"8 32 128\""
    echo "  export row_shuffle_seeds=\"40 41 42\""
    echo "  export model_name=\"openai::gpt-4o\""
    echo "  export device_id=\"1\"  # Optional: specify GPU device (default: \"0\")"
    echo "  # ... other parameters"
}

# Check if required parameters are set
if [ -z "$dataset_names" ] || [ -z "$train_chunk_sizes" ] || [ -z "$row_shuffle_seeds" ]; then
    echo "Error: Required parameters not set!"
    echo ""
    print_usage
    exit 1
fi

# Define supported ML models (must match ml_utils.py)
ml_models=("randomforest" "rf" "knn" "xgboost" "xgb")

# Function to check if model_name is an ML model
is_ml_model() {
    local model="$1"
    model_lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
    for ml_model in "${ml_models[@]}"; do
        if [[ "$model_lower" == "$ml_model" ]]; then
            return 0  # true
        fi
    done
    return 1  # false
}

# Determine which prediction script to use
if is_ml_model "$model_name"; then
    prediction_script="./src/evaluation/model_pred/ml_model_pred.py"
    script_type="ML"
    echo "🤖 Detected ML model: $model_name -> Using ml_model_pred.py"
else
    prediction_script="./src/evaluation/model_pred/dl_model_pred.py"
    script_type="DL"
    echo "🧠 Detected DL model: $model_name -> Using dl_model_pred.py"
fi

# Convert space-separated strings to arrays
read -ra DATASETS <<< "$dataset_names"
read -ra CHUNK_SIZES <<< "$train_chunk_sizes"
read -ra SHUFFLE_SEEDS <<< "$row_shuffle_seeds"

# Print configuration
echo "=============================================================================="
echo "Optimized Batch Model Prediction Configuration"
echo "=============================================================================="
echo "Script type: $script_type ($prediction_script)"
echo "Prompt data directory: $prompt_data_dir"
echo "Predict data directory: $predict_data_dir"
echo "Datasets: ${DATASETS[*]}"
echo "Train chunk sizes: ${CHUNK_SIZES[*]}"
echo "Row shuffle seeds: ${SHUFFLE_SEEDS[*]}"
echo "Split seed: $split_seed"
echo "Test chunk size: $test_chunk_size"
echo "Model name: $model_name"
echo "Temperature: $temperature"
echo "Max samples: ${max_samples:-\"(no limit)\"}"
echo "Max workers: ${max_workers:-\"(auto)\"}"
echo "Labels: ${labels:-\"(auto-detect)\"}"
echo "Device ID: $device_id"
echo "Force overwrite: $force_overwrite"
echo "Logprobs supported: $logprobs_supported"
echo "🎯 Optimization: Model loaded once per chunk_size!"
echo "=============================================================================="
echo ""

# Counter for tracking progress
total_chunk_sizes=${#CHUNK_SIZES[@]}
total_combinations_per_chunk=$((${#DATASETS[@]} * ${#SHUFFLE_SEEDS[@]}))
total_combinations=$((total_chunk_sizes * total_combinations_per_chunk))
current_chunk=0

echo "📊 Processing Summary:"
echo "   - ${#CHUNK_SIZES[@]} chunk sizes"
echo "   - ${#DATASETS[@]} datasets × ${#SHUFFLE_SEEDS[@]} seeds = $total_combinations_per_chunk combinations per chunk"
echo "   - Total combinations: $total_combinations"
echo "   - Model loads: ${#CHUNK_SIZES[@]} times (instead of $total_combinations times!)"
echo ""

# Optimized processing loop: chunk_size -> load model -> (datasets × seeds)
for train_chunk_size in "${CHUNK_SIZES[@]}"; do
    current_chunk=$((current_chunk + 1))
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔧 Processing Chunk Size: $train_chunk_size ($current_chunk/${#CHUNK_SIZES[@]})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 Model will be loaded once for this chunk size and process $total_combinations_per_chunk combinations"
    echo ""
    
    # Build the command based on script type
    if [[ "$script_type" == "ML" ]]; then
        # ML model prediction - uses 1_split as input, different parameters
        cmd="python $prediction_script \
            --input_dir \"./w1/bucket_1/1_split\" \
            --output_dir \"$predict_data_dir\" \
            --dataset_names \"${DATASETS[*]}\" \
            --row_shuffle_seeds \"${SHUFFLE_SEEDS[*]}\" \
            --split_seed \"$split_seed\" \
            --train_chunk_size \"$train_chunk_size\" \
            --test_chunk_size \"$test_chunk_size\" \
            --model_name \"$model_name\""
        
        # Add ML-specific optional parameters
        if [ -n "$max_samples" ]; then
            cmd="$cmd --max_samples \"$max_samples\""
        fi
        
        if [ -n "$max_workers" ]; then
            cmd="$cmd --max_workers \"$max_workers\""
        fi
        
        if [ "$force_overwrite" = "True" ] || [ "$force_overwrite" = "true" ]; then
            cmd="$cmd --force_overwrite"
        fi
        
    else
        # DL model prediction - uses 2_prompt as input, different parameters
        cmd="python $prediction_script \
            --input_dir \"$prompt_data_dir\" \
            --output_dir \"$predict_data_dir\" \
            --dataset_names \"${DATASETS[*]}\" \
            --row_shuffle_seeds \"${SHUFFLE_SEEDS[*]}\" \
            --split_seed \"$split_seed\" \
            --train_chunk_size \"$train_chunk_size\" \
            --test_chunk_size \"$test_chunk_size\" \
            --model_name \"$model_name\" \
            --temperature \"$temperature\" \
            --device_id \"$device_id\""
        
        # Add DL-specific optional parameters
        if [ -n "$max_samples" ]; then
            cmd="$cmd --max_samples \"$max_samples\""
        fi
        
        if [ -n "$labels" ]; then
            cmd="$cmd --labels \"$labels\""
        fi
        
        if [ "$force_overwrite" = "True" ] || [ "$force_overwrite" = "true" ]; then
            cmd="$cmd --force_overwrite"
        fi
        
        # Add logprobs_supported parameter for DL models
        if [ "$logprobs_supported" = "True" ] || [ "$logprobs_supported" = "true" ]; then
            cmd="$cmd --logprobs-supported"
        else
            cmd="$cmd --no-logprobs-supported"
        fi
    fi
    
    echo "💡 Command: $cmd"
    echo ""
    
    # Run the actual command
    eval $cmd
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully processed chunk_size $train_chunk_size ($total_combinations_per_chunk combinations)"
    else
        echo "❌ Failed to process chunk_size $train_chunk_size"
        echo "Error code: $?"
    fi
    echo "==================================================================================="
    echo ""
done

echo "🎉 Optimized batch model prediction completed!"
echo "📊 Performance Summary:"
echo "   - Script type: $script_type"
echo "   - Processed ${#CHUNK_SIZES[@]} chunk sizes"
echo "   - Total combinations: $total_combinations"
echo "   - Model loads: ${#CHUNK_SIZES[@]} (saved $((total_combinations - ${#CHUNK_SIZES[@]})) model loads! 🚀)"
saved_loads=$((total_combinations - ${#CHUNK_SIZES[@]}))
efficiency_gain=$((100 * saved_loads / total_combinations))
echo "   - Efficiency gain: ~${efficiency_gain}% reduction in model loading overhead"
