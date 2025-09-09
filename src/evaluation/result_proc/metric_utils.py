"""
Majority Voting Evaluator Utility Functions
"""

import os
import json
import re
from collections import defaultdict
import numpy as np


def parse_and_unify_json(response_str, prob_mapping=None, default_label=0):
    """
    Parse and unify JSON response format
    
    Args:
        response_str: Raw response string
        prob_mapping: Probability mapping dictionary {item_id: {"0": prob0, "1": prob1, ...}}
        default_label: Default label to use when unable to infer label
        
    Returns:
        list: List of unified prediction results
    """
    response_str = response_str.replace("test_id", "id")
    cleaned = re.sub(r'\r\n|\r', '\n', response_str.strip())
    cleaned = re.sub(r'[\x00-\x1F]+', ' ', cleaned)
    
    # Fix malformed JSON arrays
    cleaned = re.sub(r'\[\s*"([^"]+)"\s*:\s*([^,\]]+)\s*,\s*"([^"]+)"\s*:\s*([^,\]]+)\s*\]', 
                    r'{"\\1": \\2, "\\3": \\4}', cleaned)
    
    try:
        parsed_json = json.loads(cleaned.replace("'", '"'))
    except json.JSONDecodeError:
        json_candidates = []
        stack = []
        start_idx = -1
        
        for i, char in enumerate(cleaned):
            if char in '{[':
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char in '}]':
                if stack:
                    stack.pop()
                    if not stack and start_idx != -1:
                        json_candidates.append(cleaned[start_idx:i + 1])
                        start_idx = -1
        
        parsed_json = None
        for candidate in sorted(json_candidates, key=len, reverse=True):
            try:
                candidate = re.sub(r'\[\s*"([^"]+)"\s*:\s*([^,\]]+)\s*,\s*"([^"]+)"\s*:\s*([^,\]]+)\s*\]', 
                                 r'{"\\1": \\2, "\\3": \\4}', candidate)
                candidate = re.sub(r'(?m)^\s*(//|#).*?\n', '', candidate)
                candidate = re.sub(r'/\*.*?\*/', '', candidate, flags=re.DOTALL)
                candidate = re.sub(r',\s*]', ']', candidate)
                candidate = re.sub(r',\s*}', '}', candidate)
                parsed_json = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        
        if parsed_json is None:
            raise ValueError("Unparseable JSON format")
    
    if not isinstance(parsed_json, list):
        parsed_json = [parsed_json] if parsed_json is not None else []
    
    unified = []
    for idx, item in enumerate(parsed_json):
        if not isinstance(item, dict):
            item = {"label": str(item)}

        item_id = str(item["id"])
        
        try:
            label = int(item.get("label"))
            unified.append({"id": item_id, "label": label})
        except (ValueError, TypeError):
            # Invalid label, try to infer from probabilities
            if prob_mapping and item_id in prob_mapping:
                label_probs = prob_mapping[item_id]
                # Find label with highest probability
                best_label = None
                best_prob = -1
                
                for label_str, prob in label_probs.items():
                    if prob > best_prob:
                        best_prob = prob
                        best_label = label_str
                
                if best_label is not None:
                    try:
                        inferred_label = int(best_label)
                        unified.append({"id": item_id, "label": inferred_label})
                        print(f"🔄 Inferred label: item_id={item_id}, original='{item.get('label')}' → {inferred_label} (prob={best_prob:.3f})")
                        continue
                    except (ValueError, TypeError):
                        pass
            
            # If probability inference fails, use provided default label instead of skipping
            print(f"⚠️  Invalid label using default: item_id={item_id}, original='{item.get('label')}' → {default_label} (unable to infer)")
            unified.append({"id": item_id, "label": default_label})
    
    return unified


def extract_model_prefix(model_name):
    """
    Extract model name prefix for file naming
    
    Rules:
    1. For `openai::` formats, use the part after the prefix
    2. If `/` is present, use only the last segment (e.g. `minzl/toy_3550` → `toy_3550`)
    3. Use original name in all other cases
    
    Args:
        model_name: Full model name
        
    Returns:
        str: Model prefix for file naming
    """
    if '::' in model_name:
        # Handle backend::model format
        backend, actual_model = model_name.split('::', 1)
        if backend.lower() == 'openai':
            # For openai::model, use the actual model name
            model_name = actual_model
        else:
            # For other backends, use full format
            model_name = model_name.replace('::', '_')
    
    # Handle HuggingFace format paths (e.g., minzl/toy_3550)
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    
    # Replace other possible special characters with underscores
    model_name = model_name.replace('-', '_').replace('.', '_')
    
    return model_name


def construct_filename(dataset_name, seed, train_chunk_size, test_chunk_size, max_samples, temperature, model_name=None):
    """
    Construct prediction filename
    
    Args:
        dataset_name: Dataset name
        seed: Random seed
        train_chunk_size: Training chunk size
        test_chunk_size: Test chunk size
        max_samples: Maximum samples (compatibility parameter, ignored)
        temperature: Temperature parameter (compatibility parameter, ignored)
        model_name: Model name (optional, for new format)
        
    Returns:
        str: Constructed filename
    """
    if model_name:
        # New format: model_prefix@@dataset_Rseed_trainSize_testSize.jsonl
        model_prefix = extract_model_prefix(model_name)
        return f"{model_prefix}@@{dataset_name}_Rseed{seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}.jsonl"
    else:
        # Old format: dataset_Rseed_trainSize_testSize_pred.jsonl
        return f"{dataset_name}_Rseed{seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}_pred.jsonl"


def perform_weighted_majority_voting(all_votes, all_probabilities, weighted, valid_vote_counts=None):
    """
    Perform weighted majority voting
    
    Args:
        all_votes: All voting data, format {key: [(label, weight), ...]}
                   Only contains valid votes (tag=true votes)
        all_probabilities: All probability data for AUC calculation
        weighted: Whether to use probability weighting
        valid_vote_counts: Valid vote count per key, format {key: count}
        
    Returns:
        tuple: (Voting results, Final probabilities)
    """
    result = {}
    final_probabilities = {}
    
    for key, votes_with_weights in all_votes.items():
        if not votes_with_weights:
            # No valid votes, skip this key
            continue

        # Get valid vote count for this key
        valid_count = valid_vote_counts.get(key, len(votes_with_weights)) if valid_vote_counts else len(votes_with_weights)
        
        if weighted:
            # Probability-weighted voting: sum probabilities for each label, then normalize by valid vote count
            label_weights = defaultdict(float)
            for label, weight in votes_with_weights:
                label_weights[label] += weight
            
            # Optional: Normalize by valid vote count (depending on requirements)
            # Here we keep cumulative logic since probability weighting already considers weights
            
        else:
            # Equal-weight voting: count votes for each label, consider valid vote count
            label_weights = defaultdict(int)
            for label, weight in votes_with_weights:
                label_weights[label] += 1
            
            # For equal-weight voting, we can calculate percentages
            # But here we still use absolute counts to determine winning label

        if label_weights:
            # Get predicted label (highest total weight/count)
            predicted_label = max(label_weights.items(), key=lambda x: x[1])[0]
            result[key] = predicted_label
            
            # Output voting details (for debugging)
            total_weight = sum(label_weights.values())
            winning_weight = label_weights[predicted_label]
            confidence_ratio = winning_weight / total_weight if total_weight > 0 else 0
            
            # Only output detailed info for a few keys to avoid excessive logs
            if len(result) <= 3:  # Show details for first 3 keys only
                sorted_labels = sorted(label_weights.items(), key=lambda x: x[1], reverse=True)
                print(f"🗳️  Key {key}: valid_votes={valid_count}, vote_distribution={dict(sorted_labels)}, winning_label={predicted_label} (confidence={confidence_ratio:.3f})")
            
            # Calculate average probability for positive class
            if key in all_probabilities:
                prob_votes = all_probabilities[key]
                if prob_votes:
                    final_probabilities[key] = sum(prob_votes) / len(prob_votes)
                else:
                    final_probabilities[key] = 0.5
            else:
                final_probabilities[key] = 0.5

    return result, final_probabilities


def parse_classification_report_to_json(report, base_metadata, auc_score):
    """
    Generic classification report parsing function, suitable for both single-file and batch evaluation
    
    Args:
        report: sklearn classification report string
        base_metadata: Base metadata dictionary
        auc_score: AUC score
        
    Returns:
        dict: Parsed JSON data
    """
    # Copy base metadata
    json_data = base_metadata.copy()
    json_data["AUC Score"] = auc_score if auc_score is not None else "Not calculated (requires binary classification)"
    
    # Parse classification report
    lines = report.strip().split('\n')
    
    # Extract metrics for each class
    precision_values = []
    recall_values = []
    f1_values = []
    support_values = []
    
    # Find table data rows
    for line in lines:
        line = line.strip()
        if line.startswith('Class '):
            # Parse class line, e.g.: "Class 0     0.5000    0.5763    0.5354        59"
            parts = line.split()
            if len(parts) >= 5:
                precision_values.append(float(parts[2]))
                recall_values.append(float(parts[3]))
                f1_values.append(float(parts[4]))
                support_values.append(int(parts[5]))
    
    # Add detailed metrics - grouped by class
    class_metrics = {}
    for i in range(len(precision_values)):
        class_metrics[f"class_{i}"] = {
            "precision": precision_values[i],
            "recall": recall_values[i],
            "f1-score": f1_values[i],
            "support": support_values[i]
        }
    
    json_data["class_metrics"] = class_metrics
    
    # Extract accuracy, macro avg, weighted avg
    for line in lines:
        line = line.strip()
        if line.startswith('accuracy'):
            # e.g.: "accuracy                         0.4891       640"
            parts = line.split()
            if len(parts) >= 2:
                json_data["accuracy"] = float(parts[1])
        elif line.startswith('macro avg'):
            # e.g.: "macro avg     0.4820    0.4911    0.4831       640"
            parts = line.split()
            if len(parts) >= 5:
                json_data["macro_avg_precision"] = float(parts[2])
                json_data["macro_avg_recall"] = float(parts[3])
                json_data["macro_avg_f1"] = float(parts[4])
                json_data["macro_avg_support"] = int(parts[5])
        elif line.startswith('weighted avg'):
            # e.g.: "weighted avg     0.4862    0.4891    0.4839       640"
            parts = line.split()
            if len(parts) >= 5:
                json_data["weighted_avg_precision"] = float(parts[2])
                json_data["weighted_avg_recall"] = float(parts[3])
                json_data["weighted_avg_f1"] = float(parts[4])
                json_data["weighted_avg_support"] = int(parts[5])
    
    return json_data


def _parse_classification_report(report, dataset_name, model_name, voting_method, 
                               row_shuffle_seeds, 
                               total_combinations, processed_combinations,
                               train_chunk_size, test_chunk_size, 
                               bad_sample_count, auc_score):
    """
    Parse classification report and generate JSON data (batch voting version)
    """
    # Build batch voting base metadata
    base_metadata = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Voting Method": voting_method,
        "Default Label Strategy": "Training Data Most Frequent",
        "Random Seeds": row_shuffle_seeds,
        "Total combinations processed": total_combinations,
        "Processed combinations (seeds)": processed_combinations,
        "Chunk sizes": f"train={train_chunk_size}, test={test_chunk_size}",
        "Responses with wrong sample size": bad_sample_count
    }
    
    # Use generic parsing function
    return parse_classification_report_to_json(report, base_metadata, auc_score)


def save_single_file_results(output_txt_file, input_jsonl_file, dataset_name, 
                           report, auc_score, total_samples, bad_sample_count):
    """
    Save single file evaluation results (TXT + JSON)
    
    Args:
        output_txt_file: Output TXT file path
        input_jsonl_file: Input JSONL file path
        dataset_name: Dataset name
        report: Classification report string
        auc_score: AUC score
        total_samples: Total samples
        bad_sample_count: Bad sample count
    """
    # Build single file evaluation base metadata
    base_metadata = {
        "Dataset": dataset_name,
        "Input file": input_jsonl_file,
        "Total samples": total_samples,
        "Responses with wrong sample size": bad_sample_count,
        "Evaluation mode": "Direct single file assessment"
    }
    
    # Use generic parsing function to generate JSON data
    json_data = parse_classification_report_to_json(report, base_metadata, auc_score)
    
    # Save TXT file
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        f.write(f"[AML Version] Single File Evaluation Report:\n\n")
        f.write(f"Input file: {input_jsonl_file}\n")
        f.write(f"Total samples processed: {total_samples}\n")
        f.write(f"Responses with wrong sample size: {bad_sample_count}\n\n")
        f.write(report)
        if auc_score is not None:
            f.write(f"\n[AML Version] AUC Score: {auc_score:.4f}\n")
    
    # Intelligently generate JSON file path
    if output_txt_file.endswith('.txt'):
        json_path = output_txt_file.replace('.txt', '.json')
    elif output_txt_file.endswith('.json'):
        # If user specified .json file, we need to generate corresponding .txt file
        json_path = output_txt_file
        txt_path = output_txt_file.replace('.json', '.txt')
        # Re-save TXT file to correct path
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"[AML Version] Single File Evaluation Report:\n\n")
            f.write(f"Input file: {input_jsonl_file}\n")
            f.write(f"Total samples processed: {total_samples}\n")
            f.write(f"Responses with wrong sample size: {bad_sample_count}\n\n")
            f.write(report)
            if auc_score is not None:
                f.write(f"\n[AML Version] AUC Score: {auc_score:.4f}\n")
        print(f"✅ Results saved to: {txt_path}")
    else:
        # If no extension, default add .json
        json_path = output_txt_file + '.json'
    
    # Save JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
        
    if not output_txt_file.endswith('.json'):
        print(f"✅ Results saved to: {output_txt_file}")
    print(f"✅ JSON data saved to: {json_path}")


def save_results(result_output_dir, dataset_name, model_name, row_shuffle_seeds,
                train_chunk_size, test_chunk_size, weighted,
                report, auc_score, processed_combinations, bad_sample_count):
    """
    Save evaluation results
    """
    seeds_str = '_'.join(map(str, row_shuffle_seeds))
    config_str = f"seeds{seeds_str}"
    
    if weighted:
        voting_type = "probability_weighted_vote"
        voting_method = "Probability-Weighted Majority Voting"
    else:
        voting_type = "equal_weight_vote"
        voting_method = "Equal-Weight Majority Voting"
        
    filename = f"{dataset_name}_{voting_type}_trainSize{train_chunk_size}_{config_str}.txt"
    result_file = os.path.join(result_output_dir, filename)

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Voting Method: {voting_method}\n")
        f.write(f"Default Label Strategy: Training Data Most Frequent\n")
        f.write(f"Random Seeds: {row_shuffle_seeds}\n")
        f.write(f"Total combinations processed: {len(processed_combinations)}\n")
        f.write(f"Processed combinations (seeds): {processed_combinations}\n")
        f.write(f"Chunk sizes: train={train_chunk_size}, test={test_chunk_size}\n")
        f.write(f"Responses with wrong sample size: {bad_sample_count}\n")
        
        if auc_score is not None:
            f.write(f"AUC Score: {auc_score:.4f}\n")
        else:
            f.write("AUC Score: Not calculated (requires binary classification)\n")
            
        f.write("-" * 70 + "\n\n")
        f.write(report)

    # Parse classification report and generate JSON file
    base_filename = f"{dataset_name}_{voting_type}_trainSize{train_chunk_size}_{config_str}"
    json_file = os.path.join(result_output_dir, f"{base_filename}.json")
    
    json_data = _parse_classification_report(report, dataset_name, model_name, voting_method, 
                                            row_shuffle_seeds, 
                                            len(processed_combinations), processed_combinations,
                                            train_chunk_size, test_chunk_size, 
                                            bad_sample_count, auc_score)
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to: {result_file}")
    print(f"✅ JSON data saved to: {json_file}")