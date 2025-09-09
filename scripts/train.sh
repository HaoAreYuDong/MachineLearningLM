#!/bin/bash

cd ./third_party/LLaMA-Factory

mkdir -p logs
LOG_FILE="logs/train_$(date +'%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "====================================="
echo "📝 All output will be logged to: $LOG_FILE"
echo "====================================="


combinations=(
  # "$MODEL" "$TEMPLATE" "$LEARNING_RATE" "$EPOCH" "$BATCH_SIZE_PER_DEVICE" "$DATASET" "$CUTOFF_LEN"  "$MAX_SAMPLES" "$LORA_RANK" "$SAVE_STEPS" "$CHECKPOINT(use "" if train from vanilla model)" "$RUN_NAME" "$OUTPUT_DIR" "$WARMUP_RATIO"
  # examples
  "Qwen/Qwen2.5-7B-Instruct" "qwen" "1e-5" "1" "1" "MachineLearningLM_Corpus_Warmup" "32000" "9999999" "8" "100" "" "warmup" "./checkpoint/warmup" "0.05"
  "Qwen/Qwen2.5-7B-Instruct" "qwen" "1e-7" "1" "1" "MachineLearningLM_Corpus_All_0" "32000" "9999999" "8" "100" "./checkpoint/warmup" "stage0" "./checkpoint/stage0" "0.05"
  "Qwen/Qwen2.5-7B-Instruct" "qwen" "1e-6" "1" "1" "MachineLearningLM_Corpus_All_1" "32000" "9999999" "8" "100" "./checkpoint/stage0" "stage1" "./checkpoint/stage1" "0.05"
  
)

total_combos=${#combinations[@]}
combo_size=14
#[base_model, template, lr, epoch, batch_size, dataset, cutoff_len, max_samples,lora_rank,save_steps,resume_from_checkpoint,run_name,output_dir,warmup_ratio]

if [ $((total_combos % combo_size)) -ne 0 ]; then
  echo "❌ Error: Invalid number of parameters in combinations array"
  exit 1
fi

for ((i=0; i<total_combos; i+=combo_size)); do
  base_model="${combinations[i]}"
  template="${combinations[i+1]}"
  lr="${combinations[i+2]}"
  epoch="${combinations[i+3]}"
  batch_size="${combinations[i+4]}"
  dataset="${combinations[i+5]}"
  cutoff_len="${combinations[i+6]}"
  max_samples="${combinations[i+7]}"
  lora_rank="${combinations[i+8]}"
  save_steps="${combinations[i+9]}"
  checkpoint="${combinations[i+10]}"
  run_name="${combinations[i+11]}"
  output_dir="${combinations[i+12]}"
  warmup_ratio="${combinations[i+13]}"
 
  model_name=$(basename "$base_model")
  

  model_id="${model_name}_${dataset}_MULTINODE"
  model_id=$(echo "$model_id" | sed 's/[^a-zA-Z0-9_-]/_/g')
  
  echo "=================================================================="
  echo "🚀 Starting training with Base Model: $model_name"
  echo "⚙️  Template: $template, LR: $lr, Epochs: $epoch, Batch Size: $batch_size, Dataset: $dataset, cutoff_len:$cutoff_len, max_samples: $max_samples, lora_rank:$lora_rank, save_steps:$save_steps,checkpoint:$checkpoint,run_name:$run_name, warmup_ratio:$warmup_ratio"
  echo "📁 Model ID: $model_id"
  echo "=================================================================="
  

  mkdir -p "$output_dir"
  
  
  llamafactory-cli train examples/train_lora/qwen2.5_lora_sft.yaml \
    model_name_or_path="$base_model" \
    template="$template" \
    dataset="$dataset" \
    learning_rate="$lr" \
    num_train_epochs="$epoch" \
    per_device_train_batch_size="$batch_size" \
    output_dir="$output_dir" \
    cutoff_len="$cutoff_len" \
    max_samples="$max_samples" \
    lora_rank="$lora_rank" \
    save_steps="$save_steps" \
    adapter_name_or_path="$checkpoint" \
    run_name="$run_name" \
    output_dir="$output_dir" \
    warmup_ratio="$warmup_ratio"

  if [ $? -ne 0 ]; then
    echo "❌ Training failed for Base: $model_name, Template: $template, LR: $lr, Epochs: $epoch, Batch Size: $batch_size, Dataset: $dataset, cutoff_len:$cutoff_len, max_samples: $max_samples, lora_rank:$lora_rank, save_steps:$save_steps,checkpoint:$checkpoint,run_name:$run_name, warmup_ratio:$warmup_ratio"
    continue
  fi
  
  echo "✅ Training completed successfully!"
  echo "🔄 Proceeding to next parameter combination"
  echo "=================================================================="
  sleep 2
done

echo "✨ All specified hyperparameter combinations processed!"