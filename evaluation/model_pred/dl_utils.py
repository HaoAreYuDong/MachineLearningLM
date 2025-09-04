"""
推理工具类和函数
"""

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
    """安全地将 logprob 转换为 prob"""
    try:
        return float(math.exp(x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def to_token_str(tok_key, tokenizer):
    """确保 token key 是字符串"""
    if isinstance(tok_key, int):
        return tokenizer.convert_ids_to_tokens(tok_key, skip_special_tokens=False)
    return tok_key


def build_token_level_records(vllm_request_output, top_logprobs, tokenizer):
    """构建每个 token 的记录"""
    out = vllm_request_output.outputs[0]
    token_level = []

    # 获取采样的 token IDs
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

        # 收集所有 token 和其 logprob
        pairs = []
        for tok_key, lp_obj in lp_dict.items():
            tok_str = to_token_str(tok_key, tokenizer)
            lp = getattr(lp_obj, "logprob", lp_obj)
            pairs.append((tok_str, float(lp)))

        # 按 logprob 降序排序
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = pairs[:top_logprobs] if top_logprobs and top_logprobs > 0 else pairs

        # 确定选择的 token
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
    """从 token level 结果中提取标签概率"""
    batch_probabilities = []
    
    for i, token_record in enumerate(token_level):
        token = token_record.get("generate_token", "").strip()
        
        if token.lower() == 'label':
            # 收集 'label' 前面的连续数字 tokens
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
            
            # 查找预测的标签
            predicted_label_position = i + 3
            if predicted_label_position < len(token_level):
                predicted_token_data = token_level[predicted_label_position]
                predicted_token = predicted_token_data.get("generate_token", "").strip()
                
                # 初始化标签概率
                label_probs = {label: 0.0 for label in all_labels}
                
                # 从 topk 中提取概率
                if 'topk' in predicted_token_data and predicted_token_data['topk']:
                    for alt in predicted_token_data['topk']:
                        alt_token = alt['token'].strip()
                        if alt_token in all_labels:
                            label_probs[alt_token] = alt['prob']
                
                # 确保预测的 token 有概率
                if predicted_token in all_labels and label_probs[predicted_token] == 0.0:
                    label_probs[predicted_token] = predicted_token_data.get('prob', 0.0)
                
                result = {
                    'id': id_value,
                    'label_probs': [{'label': label, 'prob': label_probs[label]} for label in all_labels]
                }
                batch_probabilities.append(result)
    
    return batch_probabilities


def extract_probabilities_from_openai_logprobs(logprobs_data, all_labels):
    """从OpenAI的logprobs数据中提取标签概率"""
    batch_probabilities = []
    
    if not logprobs_data:
        return batch_probabilities
    
    for i, token_data in enumerate(logprobs_data):
        token = token_data['token'].strip()
        
        if token.lower() == 'label':
            # 查找ID位置在i-3
            id_value = None
            if i - 3 >= 0 and i - 3 < len(logprobs_data):
                id_token = logprobs_data[i - 3]['token'].strip()
                id_value = id_token
            
            # 查找预测标签位置在i+3
            predicted_label_position = i + 3
            if predicted_label_position < len(logprobs_data):
                predicted_token_data = logprobs_data[predicted_label_position]
                predicted_token = predicted_token_data['token'].strip()
                
                # 初始化标签概率
                label_probs = {label: 0.0 for label in all_labels}
                
                # 从top_logprobs中提取概率
                if 'top_logprobs' in predicted_token_data and predicted_token_data['top_logprobs']:
                    for alt in predicted_token_data['top_logprobs']:
                        alt_token = alt['token'].strip()
                        if alt_token in all_labels:
                            # 将logprob转换为概率
                            alt_logprob = alt.get('logprob', float('-inf'))
                            label_probs[alt_token] = exp_safe(alt_logprob)
                
                # 确保预测的token有概率
                if predicted_token in all_labels and label_probs[predicted_token] == 0.0:
                    # 将logprob转换为概率
                    predicted_logprob = predicted_token_data.get('logprob', float('-inf'))
                    label_probs[predicted_token] = exp_safe(predicted_logprob)
                
                result = {
                    'id': id_value,
                    'label_probs': [{'label': label, 'prob': label_probs[label]} for label in all_labels]
                }
                batch_probabilities.append(result)
    
    return batch_probabilities


class BaseRunner(ABC):
    """推理器抽象基类"""
    
    def __init__(self, model_name, temperature=0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = 1024
        self.top_logprobs = 10  # 统一的top_logprobs设置
    
    @abstractmethod
    def generate(self, prompts):
        """生成文本，返回结果列表"""
        pass
    
    def extract_labels_from_label_info(self, jsonl_file):
        """按优先级顺序查找 label_transform_info.json 读取标签"""
        jsonl_dir = os.path.dirname(jsonl_file)
        
        # 候选路径列表，按优先级排序
        candidate_paths = []
        
        # 1. 首先在同目录下查找
        candidate_paths.append(os.path.join(jsonl_dir, "label_transform_info.json"))
        
        # 2. 如果路径包含 2_prompt，尝试替换为 1_split
        if "2_prompt" in jsonl_dir:
            split_dir = jsonl_dir.replace("2_prompt", "1_split")
            candidate_paths.append(os.path.join(split_dir, "label_transform_info.json"))
        
        # 按优先级尝试每个候选路径
        for i, label_info_file in enumerate(candidate_paths):
            if os.path.exists(label_info_file):
                try:
                    with open(label_info_file, 'r', encoding='utf-8') as f:
                        label_info = json.load(f)
                    
                    # 从 label_mapping 中提取所有标签值
                    if 'label_mapping' in label_info:
                        labels = list(label_info['label_mapping'].values())
                        # 转换为字符串并排序
                        sorted_labels = sorted([str(label) for label in labels], key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else float('inf'), x))
                        print(f"INFO: Extracted labels from label_transform_info.json (priority {i+1}): {sorted_labels}")
                        print(f"OUTPUT: Found at: {label_info_file}")
                        return sorted_labels
                    else:
                        print(f"WARNING:  No 'label_mapping' found in {label_info_file}")
                        continue
                except Exception as e:
                    print(f"WARNING:  Error reading {label_info_file}: {e}")
                    continue
        
        # 3. 如果前两个优先级都失败，尝试从 1_split 数据文件夹中的CSV文件提取标签
        if "2_prompt" in jsonl_dir:
            split_dir = jsonl_dir.replace("2_prompt", "1_split")
            try:
                # 查找 Rseed 文件夹
                if os.path.exists(split_dir):
                    rseed_dirs = [d for d in os.listdir(split_dir) if d.startswith('Rseed') and os.path.isdir(os.path.join(split_dir, d))]
                    if rseed_dirs:
                        # 选择第一个 Rseed 文件夹
                        rseed_dir = os.path.join(split_dir, rseed_dirs[0])
                        print(f"OUTPUT: Trying to extract labels from CSV files in: {rseed_dir}")
                        
                        # 查找 y_train 或 y_test CSV 文件
                        for csv_name in ['y_train.csv', 'y_test.csv']:
                            csv_path = os.path.join(rseed_dir, csv_name)
                            if os.path.exists(csv_path):
                                try:
                                    df = pd.read_csv(csv_path)
                                    
                                    # 获取所有唯一标签值
                                    if len(df.columns) > 0:
                                        # 取第一列作为标签列
                                        unique_labels = df.iloc[:, 0].unique()
                                        # 转换为字符串并排序
                                        sorted_labels = sorted([str(label) for label in unique_labels], key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else float('inf'), x))
                                        print(f"INFO: Extracted labels from CSV file (priority 3): {sorted_labels}")
                                        print(f"OUTPUT: Found at: {csv_path}")
                                        return sorted_labels
                                except Exception as e:
                                    print(f"WARNING:  Error reading CSV file {csv_path}: {e}")
                                    continue
            except Exception as e:
                print(f"WARNING:  Error accessing 1_split directory: {e}")
        
        print(f"WARNING:  label_transform_info.json not found in any candidate locations:")
        for path in candidate_paths:
            print(f"     - {path}")
        print(f"WARNING:  Also tried extracting from CSV files in 1_split data folders")
        return None

    def extract_labels_from_jsonl(self, jsonl_file):
        all_labels = set()
        
        with open(jsonl_file, 'r', encoding='utf-8') as fin:
            for line in tqdm(fin, desc="Extracting labels from JSONL"):
                data = json.loads(line.strip())
                msgs = data["messages"]
                
                # 方法1: 从user prompt中提取标签集合
                user_content = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                
                # 查找 "Label set = [...]" 模式
                label_set_match = re.search(r'Label set = \[([^\]]+)\]', user_content)
                if label_set_match:
                    label_str = label_set_match.group(1)
                    # 解析标签，处理数字和字符串
                    labels = [label.strip().strip('"\'') for label in label_str.split(',')]
                    all_labels.update(labels)
                
                # 方法2: 从assistant response中提取实际出现的标签
                assistant_content = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
                if assistant_content.strip():
                    try:
                        # 尝试解析JSON数组
                        response_data = json.loads(assistant_content)
                        if isinstance(response_data, list):
                            for item in response_data:
                                if isinstance(item, dict) and 'label' in item:
                                    all_labels.add(str(item['label']))
                    except (json.JSONDecodeError, ValueError):
                        # 如果不是JSON格式，跳过
                        pass
        
        # 转换为排序后的列表，数字优先排序
        sorted_labels = sorted(list(all_labels), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else float('inf'), x))
        print(f"INFO: Extracted labels from JSONL: {sorted_labels}")
        return sorted_labels

    def process_file(self, input_jsonl, output_jsonl, origin_csv=None, max_samples=None, user_labels=None, force_overwrite=False):
        """处理单个文件"""
        
        # 检查输出文件是否已存在，并根据参数决定处理方式
        if os.path.exists(output_jsonl):
            # 如果设置了 force_overwrite，直接删除文件
            if force_overwrite:
                try:
                    os.remove(output_jsonl)
                    print(f"🗑️  Force overwrite enabled: removed existing file {output_jsonl}")
                except Exception as exc:
                    print(f"ERROR: Failed to remove file {output_jsonl}: {exc}")
                    return False
            else:
                # 准备多层框架警告信息
                msg_lines = [
                    "WARNING: Output file already exists.",
                    "To avoid accidental overwrite, you can choose to delete and recreate it.",
                    f"Target: {output_jsonl}",
                    "Do you want to delete this file and recreate it? (Y/N)",
                ]

                # 计算舒适的框架宽度
                inner_width = max(80, max(len(s) for s in msg_lines) + 6)
                outer_width = inner_width + 6

                # 打印嵌套框架
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

                # 检查是否为交互式环境
                
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
        
        # 如果用户提供了标签，优先使用用户标签
        if user_labels:
            all_labels = user_labels
            source = "user input"
        else:
            # 优先从 label_transform_info.json 提取标签，JSONL作为备选
            all_labels = self.extract_labels_from_label_info(input_jsonl)
            if all_labels:
                source = "label_transform_info.json"
            else:
                print("WARNING:  Falling back to extracting labels from JSONL content")
                try:
                    all_labels = self.extract_labels_from_jsonl(input_jsonl)
                    if not all_labels:
                        raise ValueError("No labels found in JSONL file")
                    source = "JSONL content"
                except Exception as e:
                    print(f"ERROR: Error: Could not extract labels from JSONL: {e}")
                    return False
        
        print(f"INFO: Dataset: {Path(input_jsonl).stem}, Labels from {source}: {all_labels} (total: {len(all_labels)})")
        
        # 添加明显的最终标签确认打印
        print(f"TARGET: FINAL LABELS TO USE: {all_labels}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
        
        # 加载 prompts
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
        
        # 运行推理
        outputs = self.generate(prompts)
        
        # 保存结果
        with open(output_jsonl, 'w', encoding='utf-8') as fout:
            for idx, (output, gt) in enumerate(tqdm(zip(outputs, answers), desc="Saving results")):
                # 根据不同的推理器类型处理输出
                if hasattr(output, 'text') and hasattr(output, 'logprobs_data'):  # OpenAI 结果对象
                    generated_text = output.text
                    # 使用统一的OpenAI logprobs数据处理
                    if hasattr(self, 'logprobs_supported') and self.logprobs_supported and output.logprobs_data:
                        label_probabilities = extract_probabilities_from_openai_logprobs(output.logprobs_data, all_labels)
                    else:
                        label_probabilities = []
                elif hasattr(output, 'outputs') and hasattr(output.outputs[0], 'text'):  # vLLM输出
                    generated_text = output.outputs[0].text
                    # 构建 token level 记录
                    if hasattr(self, 'logprobs_supported') and self.logprobs_supported:
                        token_level = build_token_level_records(output, getattr(self, 'top_logprobs', 10), getattr(self, 'tokenizer', None))
                        label_probabilities = extract_probabilities_from_token_level(token_level, all_labels)
                    else:
                        label_probabilities = []
                else:  # 纯文本输出（普通OpenAI）
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
        
        print(f"SUCCESS: Completed: {output_jsonl}")
        return True
