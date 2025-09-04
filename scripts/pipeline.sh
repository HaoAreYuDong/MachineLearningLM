#!/bin/bash

# =============================================================================
# Automated End-to-End Machine Learning Evaluation Pipeline
# =============================================================================
# This script combines all the mp/ scripts into a single automated workflow:
# 1. Data Preparation (parallel processing)
# 2. Prompt Generation (parallel processing)
# 3. Model Prediction (single GPU processing)
# 4. Evaluation (parallel processing)
# 5. Report Generation (summary processing)
# 
# Usage:
#   source ./params.sh
#   ./server_pipeline.sh
# =============================================================================

# Function to print colored output
print_header() {
    echo "=============================================================================="
    echo "🚀 $1"
    echo "=============================================================================="
}

print_step() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 Step $1: $2"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

print_success() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1"
}

# Check if parameters are loaded
if [ -z "$dataset_names" ] || [ -z "$train_chunk_sizes" ] || [ -z "$row_shuffle_seeds" ]; then
    print_error "Parameters not loaded! Please run: source ./params.sh"
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

# Determine model type
if is_ml_model "$model_name"; then
    model_type="ML"
    prediction_script="./evaluation/model_pred/ml_model_pred.py"
    # ML models use 1_split as input
    model_input_dir=${model_input_dir:-"${datahub_outputs_dir:-./datahub_outputs}/1_split"}
elif [[ "$model_name" == openai::* ]]; then
    model_type="API"
    prediction_script="./evaluation/model_pred/dl_model_pred.py"
    # API models use 2_prompt as input
    model_input_dir=${model_input_dir:-"${datahub_outputs_dir:-./datahub_outputs}/2_prompt"}
else
    model_type="GPU"
    prediction_script="./evaluation/model_pred/dl_model_pred.py"
    # GPU models use 2_prompt as input
    model_input_dir=${model_input_dir:-"${datahub_outputs_dir:-./datahub_outputs}/2_prompt"}
fi

# Store start time
start_time=$(date +%s)

print_header "Automated Machine Learning Evaluation Pipeline"
echo "📊 Configuration:"
echo "   - Model Type: $model_type ($prediction_script)"
echo "   - Datasets: $dataset_names"
echo "   - Chunk sizes: $train_chunk_sizes"
echo "   - Row shuffle seeds: $row_shuffle_seeds"
echo "   - Model: $model_name"
echo "   - GPU device: $device_id"
echo "   - Max parallel jobs: $max_parallel_jobs"
echo ""

# Convert space-separated strings to arrays for summary
read -ra DATASETS <<< "$dataset_names"
read -ra CHUNK_SIZES <<< "$train_chunk_sizes"
read -ra SHUFFLE_SEEDS <<< "$row_shuffle_seeds"

total_configs_per_dataset=$((${#CHUNK_SIZES[@]} * ${#SHUFFLE_SEEDS[@]}))
total_configs=$((${#DATASETS[@]} * total_configs_per_dataset))

echo "📈 Processing Summary:"
echo "   - ${#DATASETS[@]} datasets"
echo "   - ${#CHUNK_SIZES[@]} chunk sizes × ${#SHUFFLE_SEEDS[@]} shuffle seeds = $total_configs_per_dataset configs per dataset"
echo "   - Total configurations: $total_configs"
echo ""

# =============================================================================
# STEP 1: DATA PREPARATION
# =============================================================================
print_step "1" "Data Preparation (Multi-Processing)"

# Default parameters for data prep
# Prefer configuration from params.sh; fall back to sensible defaults
datahub_outputs_dir=${datahub_outputs_dir:-"./datahub_outputs"}
original_data_dir=${original_data_dir:-"./datahub_inputs/data_raw"}
split_data_dir=${split_data_dir:-"$datahub_outputs_dir/1_split"}
split_seed=${split_seed:-42}
test_chunk_size=${test_chunk_size:-50}
test_size=${test_size:-0.2}
shuffle_columns=${shuffle_columns:-True}
max_workers=${max_workers:-2}
force_overwrite=${force_overwrite:-True}
max_parallel_jobs=${max_parallel_jobs:-6}
wait_between_datasets=${wait_between_datasets:-5}

echo "📂 Input directory: $original_data_dir"
echo "📁 Output directory: $split_data_dir"

# Function to run a single data prep configuration
run_data_prep_config() {
    local dataset=$1
    local train_chunk_size=$2
    local row_shuffle_seed=$3
    local job_id=$4
    
    echo "🚀 [Job $job_id] Data prep: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
    
    python ./evaluation/data_prep/data_chunk_prep.py \
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
        echo "✅ [Job $job_id] Data prep completed: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
    else
        echo "❌ [Job $job_id] Data prep failed: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
    fi
    
    return $exit_code
}

# Function to wait for jobs to complete
wait_for_jobs() {
    local max_jobs=$1
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 1
    done
}

# Process data preparation
overall_job_counter=0
for dataset_idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$dataset_idx]}"
    echo "Input: Processing dataset: $dataset ($(($dataset_idx + 1))/${#DATASETS[@]})"
    
    for train_chunk_size in "${CHUNK_SIZES[@]}"; do
        for row_shuffle_seed in "${SHUFFLE_SEEDS[@]}"; do
            overall_job_counter=$((overall_job_counter + 1))
            wait_for_jobs $max_parallel_jobs
            run_data_prep_config "$dataset" "$train_chunk_size" "$row_shuffle_seed" "$overall_job_counter" &
        done
    done
    
    echo "⏳ Waiting for all data prep jobs for dataset '$dataset' to complete..."
    wait
    echo "SUCCESS: Data prep completed for dataset '$dataset'"
    
    if [ $dataset_idx -lt $((${#DATASETS[@]} - 1)) ]; then
        echo "😴 Waiting ${wait_between_datasets}s before next dataset..."
        sleep $wait_between_datasets
    fi
done

print_success "Step 1: Data Preparation completed!"

# =============================================================================
# STEP 2: PROMPT GENERATION (Skip for ML models)
# =============================================================================
if [ "$model_type" = "ML" ]; then
    echo ""
    print_step "2" "Prompt Generation (Skipped for ML models)"
    echo "🤖 ML models use direct CSV data from Step 1, skipping prompt generation"
    print_success "Step 2: Prompt Generation skipped for ML models!"
else
    print_step "2" "Prompt Generation (Multi-Processing)"

    # Default parameters for prompt generation
    prompt_data_dir=${prompt_data_dir:-"$datahub_outputs_dir/2_prompt"}
    normalization=${normalization:-True}
    include_feature_descriptions=${include_feature_descriptions:-False}
    prompt_format_style=${prompt_format_style:-"concat"}

    echo "Input: Input directory: $split_data_dir"
    echo "Output: Output directory: $prompt_data_dir"

    # Function to run a single prompt generation configuration
    run_prompt_gen_config() {
        local dataset=$1
        local train_chunk_size=$2
        local row_shuffle_seed=$3
        local job_id=$4
        
        echo "STARTING: [Job $job_id] Prompt gen: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
        
        python ./evaluation/prompt_gen/data_prompt_gen.py \
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
        
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "SUCCESS: [Job $job_id] Prompt gen completed: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
        else
            echo "ERROR: [Job $job_id] Prompt gen failed: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
        fi
        
        return $exit_code
    }

    # Process prompt generation
    overall_job_counter=0
    for dataset_idx in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$dataset_idx]}"
        echo "Input: Processing dataset: $dataset ($(($dataset_idx + 1))/${#DATASETS[@]})"
        
        for train_chunk_size in "${CHUNK_SIZES[@]}"; do
            for row_shuffle_seed in "${SHUFFLE_SEEDS[@]}"; do
                overall_job_counter=$((overall_job_counter + 1))
                wait_for_jobs $max_parallel_jobs
                run_prompt_gen_config "$dataset" "$train_chunk_size" "$row_shuffle_seed" "$overall_job_counter" &
            done
        done
        
        echo "⏳ Waiting for all prompt gen jobs for dataset '$dataset' to complete..."
        wait
        echo "SUCCESS: Prompt generation completed for dataset '$dataset'"
        
        if [ $dataset_idx -lt $((${#DATASETS[@]} - 1)) ]; then
            echo "😴 Waiting ${wait_between_datasets}s before next dataset..."
            sleep $wait_between_datasets
        fi
    done

    print_success "Step 2: Prompt Generation completed!"
fi

# =============================================================================
# STEP 3: MODEL PREDICTION
# =============================================================================

# Default parameters for model prediction
predict_data_dir=${predict_data_dir:-"$datahub_outputs_dir/3_predict"}
temperature=${temperature:-0.0}
max_samples=${max_samples:-""}
labels=${labels:-""}
device_id=${device_id:-"0"}

if [ "$model_type" = "ML" ]; then
    print_step "3" "Model Prediction (ML Multi-Processing)"
    echo "🤖 Using ML model: $model_name"
    echo "⚡ Enabling parallel processing for ML models"
    echo "Input: Input directory: $model_input_dir"
    echo "Output: Output directory: $predict_data_dir"
    echo "Model: Model: $model_name"
    
    # =============================================================================
    # ML Model Prediction (Parallel Processing)
    # =============================================================================
    echo "STARTING: ML models support parallel processing!"
    
    # Function to run a single prediction configuration for ML models
    run_ml_prediction_config() {
        local dataset=$1
        local train_chunk_size=$2
        local row_shuffle_seed=$3
        local job_id=$4
        
        echo "STARTING: [Job $job_id] ML prediction: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
        
        # Build the command for single configuration
        cmd="python $prediction_script \
            --input_dir \"$model_input_dir\" \
            --output_dir \"$predict_data_dir\" \
            --dataset_names \"$dataset\" \
            --row_shuffle_seeds \"$row_shuffle_seed\" \
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
        
        # Run the actual command
        eval $cmd
        
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "SUCCESS: [Job $job_id] ML prediction completed: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
        else
            echo "ERROR: [Job $job_id] ML prediction failed: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
        fi
        
        return $exit_code
    }
    
    # Process ML model prediction with parallel processing
    overall_job_counter=0
    for dataset_idx in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$dataset_idx]}"
        echo "Input: Processing dataset: $dataset ($(($dataset_idx + 1))/${#DATASETS[@]})"
        
        for train_chunk_size in "${CHUNK_SIZES[@]}"; do
            for row_shuffle_seed in "${SHUFFLE_SEEDS[@]}"; do
                overall_job_counter=$((overall_job_counter + 1))
                wait_for_jobs $max_parallel_jobs
                run_ml_prediction_config "$dataset" "$train_chunk_size" "$row_shuffle_seed" "$overall_job_counter" &
            done
        done
        
        echo "⏳ Waiting for all ML prediction jobs for dataset '$dataset' to complete..."
        wait
        echo "SUCCESS: ML prediction completed for dataset '$dataset'"
        
        if [ $dataset_idx -lt $((${#DATASETS[@]} - 1)) ]; then
            echo "😴 Waiting ${wait_between_datasets}s before next dataset..."
            sleep $wait_between_datasets
        fi
    done

elif [ "$model_type" = "API" ]; then
    print_step "3" "Model Prediction (API-based Multi-Processing)"
    echo "� Using API-based model: $model_name"
    echo "⚡ Enabling parallel processing for API calls"
    echo "Input: Input directory: $model_input_dir"
    echo "Output: Output directory: $predict_data_dir"
    echo "Model: Model: $model_name"
    
    # =============================================================================
    # API-based Model Prediction (Parallel Processing)
    # =============================================================================
    echo "STARTING: API models support parallel processing!"
    
    # Function to run a single prediction configuration for API models
    run_api_prediction_config() {
        local dataset=$1
        local train_chunk_size=$2
        local row_shuffle_seed=$3
        local job_id=$4
        
        echo "STARTING: [Job $job_id] API prediction: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
        
        # Build the command for single configuration
        cmd="python $prediction_script \
            --input_dir \"$model_input_dir\" \
            --output_dir \"$predict_data_dir\" \
            --dataset_names \"$dataset\" \
            --row_shuffle_seeds \"$row_shuffle_seed\" \
            --split_seed \"$split_seed\" \
            --train_chunk_size \"$train_chunk_size\" \
            --test_chunk_size \"$test_chunk_size\" \
            --model_name \"$model_name\" \
            --temperature \"$temperature\""
        
        # Add optional parameters if they are set
        if [ -n "$max_samples" ]; then
            cmd="$cmd --max_samples \"$max_samples\""
        fi
        
        if [ -n "$labels" ]; then
            cmd="$cmd --labels \"$labels\""
        fi
        
        if [ "$force_overwrite" = "True" ] || [ "$force_overwrite" = "true" ]; then
            cmd="$cmd --force_overwrite"
        fi
        
        # Run the actual command
        eval $cmd
        
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "SUCCESS: [Job $job_id] API prediction completed: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
        else
            echo "ERROR: [Job $job_id] API prediction failed: $dataset (chunk:$train_chunk_size, seed:$row_shuffle_seed)"
        fi
        
        return $exit_code
    }
    
    # Process API model prediction with parallel processing
    overall_job_counter=0
    for dataset_idx in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$dataset_idx]}"
        echo "Input: Processing dataset: $dataset ($(($dataset_idx + 1))/${#DATASETS[@]})"
        
        for train_chunk_size in "${CHUNK_SIZES[@]}"; do
            for row_shuffle_seed in "${SHUFFLE_SEEDS[@]}"; do
                overall_job_counter=$((overall_job_counter + 1))
                wait_for_jobs $max_parallel_jobs
                run_api_prediction_config "$dataset" "$train_chunk_size" "$row_shuffle_seed" "$overall_job_counter" &
            done
        done
        
        echo "⏳ Waiting for all API prediction jobs for dataset '$dataset' to complete..."
        wait
        echo "SUCCESS: API prediction completed for dataset '$dataset'"
        
        if [ $dataset_idx -lt $((${#DATASETS[@]} - 1)) ]; then
            echo "😴 Waiting ${wait_between_datasets}s before next dataset..."
            sleep $wait_between_datasets
        fi
    done

else
    # GPU model type
    print_step "3" "Model Prediction (Single GPU Processing)"
    echo "🎮 Using GPU-based model: $model_name"
    echo "🔧 Using single GPU optimization (device: $device_id)"
    echo "Input: Input directory: $model_input_dir"
    echo "Output: Output directory: $predict_data_dir"
    echo "Model: Model: $model_name"
    
    # =============================================================================
    # GPU-based Model Prediction (Single GPU Optimization)
    # =============================================================================
    echo "🎮 GPU models use single GPU optimization (load model once per chunk size)"
    
    # Model prediction uses optimized single GPU logic (load model once per chunk size)
    current_chunk=0
    for train_chunk_size in "${CHUNK_SIZES[@]}"; do
        current_chunk=$((current_chunk + 1))
        
        echo "🔧 Processing chunk size: $train_chunk_size ($current_chunk/${#CHUNK_SIZES[@]})"
        echo "STARTING: Loading model once for this chunk size..."
        
        # Build the command
        cmd="python $prediction_script \
            --input_dir \"$model_input_dir\" \
            --output_dir \"$predict_data_dir\" \
            --dataset_names \"${DATASETS[*]}\" \
            --row_shuffle_seeds \"${SHUFFLE_SEEDS[*]}\" \
            --split_seed \"$split_seed\" \
            --train_chunk_size \"$train_chunk_size\" \
            --test_chunk_size \"$test_chunk_size\" \
            --model_name \"$model_name\" \
            --temperature \"$temperature\" \
            --device_id \"$device_id\""
        
        # Add optional parameters if they are set
        if [ -n "$max_samples" ]; then
            cmd="$cmd --max_samples \"$max_samples\""
        fi
        
        if [ -n "$labels" ]; then
            cmd="$cmd --labels \"$labels\""
        fi
        
        if [ "$force_overwrite" = "True" ] || [ "$force_overwrite" = "true" ]; then
            cmd="$cmd --force_overwrite"
        fi
        
        # Run the actual command
        eval $cmd
        
        if [ $? -eq 0 ]; then
            echo "SUCCESS: Model prediction completed for chunk_size $train_chunk_size"
        else
            print_error "Model prediction failed for chunk_size $train_chunk_size"
            exit 1
        fi
    done
fi

print_success "Step 3: Model Prediction completed!"

# =============================================================================
# STEP 4: EVALUATION
# =============================================================================
print_step "4" "Evaluation (Multi-Processing)"

# Default parameters for evaluation
metric_data_dir=${metric_data_dir:-"$datahub_outputs_dir/4_metric"}

echo "Input: Input directory: $predict_data_dir"
echo "Output: Output directory: $metric_data_dir"

# Function to run a single evaluation configuration
run_evaluation_config() {
    local dataset=$1
    local train_chunk_size=$2
    local job_id=$3
    
    echo "STARTING: [Job $job_id] Evaluation: $dataset (chunk_size: $train_chunk_size)"
    
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
        echo "SUCCESS: [Job $job_id] Evaluation completed: $dataset (chunk_size: $train_chunk_size)"
    else
        echo "ERROR: [Job $job_id] Evaluation failed: $dataset (chunk_size: $train_chunk_size)"
    fi
    
    return $exit_code
}

# Process evaluation
overall_job_counter=0
for dataset_idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$dataset_idx]}"
    echo "Input: Processing dataset: $dataset ($(($dataset_idx + 1))/${#DATASETS[@]})"
    
    for train_chunk_size in "${CHUNK_SIZES[@]}"; do
        overall_job_counter=$((overall_job_counter + 1))
        wait_for_jobs $max_parallel_jobs
        run_evaluation_config "$dataset" "$train_chunk_size" "$overall_job_counter" &
    done
    
    echo "⏳ Waiting for all evaluation jobs for dataset '$dataset' to complete..."
    wait
    echo "SUCCESS: Evaluation completed for dataset '$dataset'"
    
    if [ $dataset_idx -lt $((${#DATASETS[@]} - 1)) ]; then
        echo "😴 Waiting ${wait_between_datasets}s before next dataset..."
        sleep $wait_between_datasets
    fi
done

print_success "Step 4: Evaluation completed!"

# =============================================================================
# STEP 5: REPORT GENERATION
# =============================================================================
print_step "5" "Report Generation"

# Default parameters for report generation
report_data_dir=${report_data_dir:-"$datahub_outputs_dir/5_report"}

echo "Input: Input directory: $metric_data_dir"
echo "Output: Output directory: $report_data_dir"

# Run metric summarizer
echo "STARTING: Generating summary report..."

python evaluation/zero_summary/metric_summarizer.py \
    --metric_data_dir "$metric_data_dir" \
    --report_data_dir "$report_data_dir" \
    --model_name "$model_name"

if [ $? -eq 0 ]; then
    print_success "Step 5: Report Generation completed!"
else
    print_error "Report generation failed!"
    exit 1
fi

# =============================================================================
# PIPELINE COMPLETION
# =============================================================================
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

print_header "Pipeline Completed Successfully! Completed:"
echo ""
echo "Config: Processing Summary:"
echo "   - ${#DATASETS[@]} datasets processed"
echo "   - ${#CHUNK_SIZES[@]} chunk sizes × ${#SHUFFLE_SEEDS[@]} shuffle seeds = $total_configs configurations"
echo "   - Model: $model_name"
echo "   - Total execution time: ${hours}h ${minutes}m ${seconds}s"
echo ""
echo "Output: Output directories:"
echo "   - Split data: $split_data_dir"
echo "   - Prompts: $prompt_data_dir"
echo "   - Predictions: $predict_data_dir"
echo "   - Metrics: $metric_data_dir"
echo "   - Reports: $report_data_dir"
echo ""
echo "Model: Pipeline optimization highlights:"
echo "   - Used parallel processing for data prep and evaluation"

if [ "$model_type" = "ML" ]; then
    echo "   - Skipped prompt generation for ML models (uses direct CSV data)"
    echo "   - Used parallel processing for ML model prediction"
    echo "   - ML models processed $total_configs configurations in parallel"
elif [ "$model_type" = "API" ]; then
    echo "   - Used parallel processing for prompt gen and API-based model prediction"
    echo "   - API models processed $total_configs configurations in parallel"
else
    echo "   - Used parallel processing for prompt gen and evaluation"
    echo "   - Optimized GPU model prediction (loaded ${#CHUNK_SIZES[@]} times instead of $total_configs times)"
    saved_loads=$((total_configs - ${#CHUNK_SIZES[@]}))
    if [ $saved_loads -gt 0 ]; then
        efficiency_gain=$((100 * saved_loads / total_configs))
        echo "   - Saved $saved_loads model loads (~${efficiency_gain}% efficiency gain)"
    fi
fi
echo ""
echo "SUCCESS: All steps completed successfully!"
