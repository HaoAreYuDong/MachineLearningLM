import os
import sys
import json
import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def exp_safe(x: float) -> float:
    try:
        return float(math.exp(x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def to_token_str(tok_key, tokenizer):
    if isinstance(tok_key, int):
        return tokenizer.convert_ids_to_tokens(tok_key, skip_special_tokens=False)
    return tok_key


def build_token_level_records(vllm_request_output, top_logprobs, tokenizer):
    out = vllm_request_output.outputs[0]
    token_level = []

    
    sampled_token_ids = getattr(out, "token_ids", None)
    sampled_tokens_str = None
    if sampled_token_ids is not None:
        sampled_tokens_str = [
            tokenizer.convert_ids_to_tokens(tid, skip_special_tokens=False)
            for tid in sampled_token_ids
        ]

    for i, lp_dict in enumerate(out.logprobs or []):
        if not lp_dict:
            continue

        pairs = []
        for tok_key, lp_obj in lp_dict.items():
            tok_str = to_token_str(tok_key, tokenizer)
            lp = getattr(lp_obj, "logprob", lp_obj)
            pairs.append((tok_str, float(lp)))

        pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = pairs[:top_logprobs] if top_logprobs and top_logprobs > 0 else pairs

        if sampled_tokens_str is not None and i < len(sampled_tokens_str):
            chosen_tok_str = sampled_tokens_str[i]
            chosen_lp = None
            for t, lp in pairs:
                if t == chosen_tok_str:
                    chosen_lp = lp
                    break
            if chosen_lp is None:
                chosen_tok_str, chosen_lp = top_pairs[0]
        else:
            chosen_tok_str, chosen_lp = top_pairs[0]

        token_level.append({
            "generate_token": chosen_tok_str,
            "logprob": float(chosen_lp),
            "prob": exp_safe(float(chosen_lp)),
            "topk": [
                {"token": t, "logprob": float(lp), "prob": exp_safe(float(lp))}
                for (t, lp) in top_pairs
            ]
        })

    return token_level


def extract_probabilities_from_token_level(token_level, all_labels):
    batch_probabilities = []
    
    for i, token_record in enumerate(token_level):
        token = token_record.get("generate_token", "").strip()
        
        if token.lower() == 'label':
            id_tokens = []
            j = i - 3
            
            while j >= 0:
                prev_token = token_level[j].get("generate_token", "").strip()
                if prev_token.isdigit():
                    id_tokens.insert(0, prev_token)
                    j -= 1
                else:
                    break
            
            id_value = "".join(id_tokens) if id_tokens else None
            
            predicted_label_position = i + 3
            if predicted_label_position < len(token_level):
                predicted_token_data = token_level[predicted_label_position]
                predicted_token = predicted_token_data.get("generate_token", "").strip()
                
                label_probs = {label: 0.0 for label in all_labels}
                
                if 'topk' in predicted_token_data and predicted_token_data['topk']:
                    for alt in predicted_token_data['topk']:
                        alt_token = alt['token'].strip()
                        if alt_token in all_labels:
                            label_probs[alt_token] = alt['prob']
                
                if predicted_token in all_labels and label_probs[predicted_token] == 0.0:
                    label_probs[predicted_token] = predicted_token_data.get('prob', 0.0)
                
                result = {
                    'id': id_value,
                    'label_probs': [{'label': label, 'prob': label_probs[label]} for label in all_labels]
                }
                batch_probabilities.append(result)
    
    return batch_probabilities


def extract_probabilities_from_openai_logprobs(logprobs_data, all_labels):
    batch_probabilities = []
    
    if not logprobs_data:
        return batch_probabilities
    
    for i, token_data in enumerate(logprobs_data):
        token = token_data['token'].strip()
        
        if token.lower() == 'label':
            id_value = None
            if i - 3 >= 0 and i - 3 < len(logprobs_data):
                id_token = logprobs_data[i - 3]['token'].strip()
                id_value = id_token
            
            predicted_label_position = i + 3
            if predicted_label_position < len(logprobs_data):
                predicted_token_data = logprobs_data[predicted_label_position]
                predicted_token = predicted_token_data['token'].strip()
                
                label_probs = {label: 0.0 for label in all_labels}
                
                if 'top_logprobs' in predicted_token_data and predicted_token_data['top_logprobs']:
                    for alt in predicted_token_data['top_logprobs']:
                        alt_token = alt['token'].strip()
                        if alt_token in all_labels:
                            alt_logprob = alt.get('logprob', float('-inf'))
                            label_probs[alt_token] = exp_safe(alt_logprob)
                
                if predicted_token in all_labels and label_probs[predicted_token] == 0.0:
                    predicted_logprob = predicted_token_data.get('logprob', float('-inf'))
                    label_probs[predicted_token] = exp_safe(predicted_logprob)
                
                result = {
                    'id': id_value,
                    'label_probs': [{'label': label, 'prob': label_probs[label]} for label in all_labels]
                }
                batch_probabilities.append(result)
    
    return batch_probabilities


class BaseRunner(ABC):
    
    def __init__(self, model_name, temperature=0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = 1024
        self.top_logprobs = 10 
    
    @abstractmethod
    def generate(self, prompts):
        pass
    
    def extract_labels_from_label_info(self, jsonl_file):
        jsonl_dir = os.path.dirname(jsonl_file)
        
        candidate_paths = []
        
        candidate_paths.append(os.path.join(jsonl_dir, "label_transform_info.json"))
        
        if "2_prompt" in jsonl_dir:
            split_dir = jsonl_dir.replace("2_prompt", "1_split")
            candidate_paths.append(os.path.join(split_dir, "label_transform_info.json"))
        
        for i, label_info_file in enumerate(candidate_paths):
            if os.path.exists(label_info_file):
                try:
                    with open(label_info_file, 'r', encoding='utf-8') as f:
                        label_info = json.load(f)
                    
                    if 'label_mapping' in label_info:
                        labels = list(label_info['label_mapping'].values())
                        sorted_labels = sorted([str(label) for label in labels], key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else float('inf'), x))
                        print(f"📊 Extracted labels from label_transform_info.json (priority {i+1}): {sorted_labels}")
                        print(f"📁 Found at: {label_info_file}")
                        return sorted_labels
                    else:
                        print(f"⚠️  No 'label_mapping' found in {label_info_file}")
                        continue
                except Exception as e:
                    print(f"⚠️  Error reading {label_info_file}: {e}")
                    continue
        
        if "2_prompt" in jsonl_dir:
            split_dir = jsonl_dir.replace("2_prompt", "1_split")
            try:
                if os.path.exists(split_dir):
                    rseed_dirs = [d for d in os.listdir(split_dir) if d.startswith('Rseed') and os.path.isdir(os.path.join(split_dir, d))]
                    if rseed_dirs:
                        rseed_dir = os.path.join(split_dir, rseed_dirs[0])
                        print(f"📁 Trying to extract labels from CSV files in: {rseed_dir}")
                        
                        for csv_name in ['y_train.csv', 'y_test.csv']:
                            csv_path = os.path.join(rseed_dir, csv_name)
                            if os.path.exists(csv_path):
                                try:
                                    df = pd.read_csv(csv_path)
                                    
                                    if len(df.columns) > 0:
                                        unique_labels = df.iloc[:, 0].unique()
                                        sorted_labels = sorted([str(label) for label in unique_labels], key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else float('inf'), x))
                                        print(f"📊 Extracted labels from CSV file (priority 3): {sorted_labels}")
                                        print(f"📁 Found at: {csv_path}")
                                        return sorted_labels
                                except Exception as e:
                                    print(f"⚠️  Error reading CSV file {csv_path}: {e}")
                                    continue
            except Exception as e:
                print(f"⚠️  Error accessing 1_split directory: {e}")
        
        print(f"⚠️  label_transform_info.json not found in any candidate locations:")
        for path in candidate_paths:
            print(f"     - {path}")
        print(f"⚠️  Also tried extracting from CSV files in 1_split data folders")
        return None

    def extract_labels_from_jsonl(self, jsonl_file):
        all_labels = set()
        
        with open(jsonl_file, 'r', encoding='utf-8') as fin:
            for line in tqdm(fin, desc="Extracting labels from JSONL"):
                data = json.loads(line.strip())
                msgs = data["messages"]
                
                user_content = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                
                label_set_match = re.search(r'Label set = \[([^\]]+)\]', user_content)
                if label_set_match:
                    label_str = label_set_match.group(1)
                    labels = [label.strip().strip('"\'') for label in label_str.split(',')]
                    all_labels.update(labels)
                
                assistant_content = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
                if assistant_content.strip():
                    try:
                        response_data = json.loads(assistant_content)
                        if isinstance(response_data, list):
                            for item in response_data:
                                if isinstance(item, dict) and 'label' in item:
                                    all_labels.add(str(item['label']))
                    except (json.JSONDecodeError, ValueError):
                        pass
        
        sorted_labels = sorted(list(all_labels), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else float('inf'), x))
        print(f"📊 Extracted labels from JSONL: {sorted_labels}")
        return sorted_labels

    def process_file(self, input_jsonl, output_jsonl, origin_csv=None, max_samples=None, user_labels=None, force_overwrite=False):
        
        if os.path.exists(output_jsonl):
            if force_overwrite:
                try:
                    os.remove(output_jsonl)
                    print(f"🗑️  Force overwrite enabled: removed existing file {output_jsonl}")
                except Exception as exc:
                    print(f"❌ Failed to remove file {output_jsonl}: {exc}")
                    return False
            else:
                msg_lines = [
                    "WARNING: Output file already exists.",
                    "To avoid accidental overwrite, you can choose to delete and recreate it.",
                    f"Target: {output_jsonl}",
                    "Do you want to delete this file and recreate it? (Y/N)",
                ]

                inner_width = max(80, max(len(s) for s in msg_lines) + 6)
                outer_width = inner_width + 6

                print("\n" + "#" * (outer_width + 4))
                print("#" + " " * (outer_width + 2) + "#")
                print("#" + "!" * (outer_width + 2) + "#")
                print("#" + " " * (outer_width + 2) + "#")

                for line in msg_lines:
                    print("#  " + "|" + line.center(inner_width) + "|  #")

                print("#" + " " * (outer_width + 2) + "#")
                print("#" + "!" * (outer_width + 2) + "#")
                print("#" + " " * (outer_width + 2) + "#")
                print("#" * (outer_width + 4) + "\n")

                
                interactive_mode = sys.stdin.isatty()
                
                if not interactive_mode:
                    print("Non-interactive environment detected: skipping file to avoid overwrite.")
                    return True
                
                try:
                    choice = input("Delete and recreate the existing file? [y/N]: ").strip().lower()
                except Exception:
                    print("No input available or input interrupted. Skipping file.")
                    return True

                if choice in ("y", "yes"):
                    try:
                        os.remove(output_jsonl)
                        print(f"Removed existing file: {output_jsonl}")
                    except Exception as exc:
                        print(f"Failed to remove file {output_jsonl}: {exc}")
                        return False
                else:
                    print("Operation aborted by user. Skipping file.")
                    return True
        
        if user_labels:
            all_labels = user_labels
            source = "user input"
        else:
            all_labels = self.extract_labels_from_label_info(input_jsonl)
            if all_labels:
                source = "label_transform_info.json"
            else:
                print("⚠️  Falling back to extracting labels from JSONL content")
                try:
                    all_labels = self.extract_labels_from_jsonl(input_jsonl)
                    if not all_labels:
                        raise ValueError("No labels found in JSONL file")
                    source = "JSONL content"
                except Exception as e:
                    print(f"❌ Error: Could not extract labels from JSONL: {e}")
                    return False
        
        print(f"📊 Dataset: {Path(input_jsonl).stem}, Labels from {source}: {all_labels} (total: {len(all_labels)})")
        
        print(f"🎯 FINAL LABELS TO USE: {all_labels}")
        
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
        
        prompts, answers = [], []
        with open(input_jsonl, 'r', encoding='utf-8') as fin:
            for i, line in enumerate(tqdm(fin, desc="Loading prompts")):
                if max_samples and i >= max_samples:
                    break
                    
                data = json.loads(line.strip())
                msgs = data["messages"]
                
                system_prompt = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
                user_prompt = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                answer = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
                
                full_prompt = f"{system_prompt}\n{user_prompt}" if system_prompt else user_prompt
                prompts.append(full_prompt)
                answers.append(answer)
        
        print(f"🧠 Starting inference on {len(prompts)} prompts")
        
        outputs = self.generate(prompts)
        
        with open(output_jsonl, 'w', encoding='utf-8') as fout:
            for idx, (output, gt) in enumerate(tqdm(zip(outputs, answers), desc="Saving results")):
                if hasattr(output, 'text') and hasattr(output, 'logprobs_data'): 
                    generated_text = output.text
                    if hasattr(self, 'logprobs_supported') and self.logprobs_supported and output.logprobs_data:
                        label_probabilities = extract_probabilities_from_openai_logprobs(output.logprobs_data, all_labels)
                    else:
                        label_probabilities = []
                elif hasattr(output, 'outputs') and hasattr(output.outputs[0], 'text'):  
                    generated_text = output.outputs[0].text
                    if hasattr(self, 'logprobs_supported') and self.logprobs_supported:
                        token_level = build_token_level_records(output, getattr(self, 'top_logprobs', 10), getattr(self, 'tokenizer', None))
                        label_probabilities = extract_probabilities_from_token_level(token_level, all_labels)
                    else:
                        label_probabilities = []
                else:  
                    generated_text = str(output)
                    label_probabilities = []
                
                record = {
                    "id": idx,
                    "response": generated_text,
                    "groundtruth": gt,
                    "batch_probabilities": label_probabilities,
                    "available_labels": all_labels
                }
                
                json.dump(record, fout, ensure_ascii=False)
                fout.write('\n')
        
        print(f"✅ Completed: {output_jsonl}")
        return True
