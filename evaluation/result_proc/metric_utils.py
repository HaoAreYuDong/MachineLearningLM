"""
多数投票评估器工具函数
"""

import os
import json
import re
from collections import defaultdict
import numpy as np


def parse_and_unify_json(response_str, prob_mapping=None, default_label=0):
    """
    解析并统一JSON响应格式
    
    Args:
        response_str: 原始响应字符串
        prob_mapping: 概率映射字典 {item_id: {"0": prob0, "1": prob1, ...}}
        default_label: 当无法推断标签时使用的默认标签
        
    Returns:
        list: 统一格式的预测结果列表
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
            # 无效标签，尝试从概率推断
            if prob_mapping and item_id in prob_mapping:
                label_probs = prob_mapping[item_id]
                # 找到概率最高的标签
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
                        print(f"🔄 推断标签: item_id={item_id}, 原标签='{item.get('label')}' → {inferred_label} (概率={best_prob:.3f})")
                        continue
                    except (ValueError, TypeError):
                        pass
            
            # 如果概率推断也失败，使用传入的默认标签而不是跳过
            print(f"WARNING:  无效标签使用默认值: item_id={item_id}, 原标签='{item.get('label')}' → {default_label} (无法推断)")
            unified.append({"id": item_id, "label": default_label})
    
    return unified


def extract_model_prefix(model_name):
    """
    提取模型名称前缀用于文件命名
    
    规则：
    1. 如果是 openai:: 格式，取后面的部分
    2. 如果包含 /，只取最后一个部分（如 minzl/toy_3550 -> toy_3550）
    3. 其他情况直接使用
    
    Args:
        model_name: 完整的模型名称
        
    Returns:
        str: 用于文件命名的模型前缀
    """
    if '::' in model_name:
        # 处理 backend::model 格式
        backend, actual_model = model_name.split('::', 1)
        if backend.lower() == 'openai':
            # 对于 openai::model，使用后面的模型名
            model_name = actual_model
        else:
            # 对于其他 backend，使用完整的格式
            model_name = model_name.replace('::', '_')
    
    # 处理 HuggingFace 格式的路径（如 minzl/toy_3550）
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    
    # 替换其他可能的特殊字符为下划线
    model_name = model_name.replace('-', '_').replace('.', '_')
    
    return model_name


def construct_filename(dataset_name, seed, train_chunk_size, test_chunk_size, max_samples, temperature, model_name=None):
    """
    构建预测文件名
    
    Args:
        dataset_name: 数据集名称
        seed: 随机种子
        train_chunk_size: 训练块大小
        test_chunk_size: 测试块大小
        max_samples: 最大样本数（兼容性参数，已忽略）
        temperature: 温度参数（兼容性参数，已忽略）
        model_name: 模型名称（可选，用于新格式）
        
    Returns:
        str: 构建的文件名
    """
    if model_name:
        # 新格式：model_prefix@@dataset_Rseed_trainSize_testSize.jsonl
        model_prefix = extract_model_prefix(model_name)
        return f"{model_prefix}@@{dataset_name}_Rseed{seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}.jsonl"
    else:
        # 旧格式：dataset_Rseed_trainSize_testSize_pred.jsonl
        return f"{dataset_name}_Rseed{seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}_pred.jsonl"


def perform_weighted_majority_voting(all_votes, all_probabilities, weighted, valid_vote_counts=None):
    """
    执行加权多数投票
    
    Args:
        all_votes: 所有投票数据，格式为 {key: [(label, weight), ...]}
                  只包含有效投票（tag为true的投票）
        all_probabilities: 所有概率数据，用于AUC计算
        weighted: 是否使用概率加权
        valid_vote_counts: 每个key的有效投票数，格式为 {key: count}
        
    Returns:
        tuple: (投票结果, 最终概率)
    """
    result = {}
    final_probabilities = {}
    
    for key, votes_with_weights in all_votes.items():
        if not votes_with_weights:
            # 没有有效投票，跳过这个key
            continue

        # 获取该key的有效投票数
        valid_count = valid_vote_counts.get(key, len(votes_with_weights)) if valid_vote_counts else len(votes_with_weights)
        
        if weighted:
            # Probability-weighted voting: sum probabilities for each label, then normalize by valid vote count
            label_weights = defaultdict(float)
            for label, weight in votes_with_weights:
                label_weights[label] += weight
            
            # 可选：按有效投票数进行归一化（取决于具体需求）
            # 这里保持累加逻辑，因为概率加权本身就考虑了权重
            
        else:
            # Equal-weight voting: count votes for each label, consider valid vote count
            label_weights = defaultdict(int)
            for label, weight in votes_with_weights:
                label_weights[label] += 1
            
            # 对于等权投票，我们可以计算百分比
            # 但这里仍然使用绝对计数来决定获胜标签

        if label_weights:
            # Get predicted label (highest total weight/count)
            predicted_label = max(label_weights.items(), key=lambda x: x[1])[0]
            result[key] = predicted_label
            
            # 输出投票详情（用于调试）
            total_weight = sum(label_weights.values())
            winning_weight = label_weights[predicted_label]
            confidence_ratio = winning_weight / total_weight if total_weight > 0 else 0
            
            # 只对少数key输出详细信息，避免过多日志
            if len(result) <= 3:  # 只显示前3个key的详情
                sorted_labels = sorted(label_weights.items(), key=lambda x: x[1], reverse=True)
                print(f"VOTE:  Key {key}: 有效投票数={valid_count}, 投票分布={dict(sorted_labels)}, 获胜标签={predicted_label} (置信度={confidence_ratio:.3f})")
            
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
    通用的分类报告解析函数，适用于单文件和批量评估
    
    Args:
        report: sklearn分类报告字符串
        base_metadata: 基础元数据字典
        auc_score: AUC分数
        
    Returns:
        dict: 解析后的JSON数据
    """
    # 复制基础元数据
    json_data = base_metadata.copy()
    json_data["AUC Score"] = auc_score if auc_score is not None else "Not calculated (requires binary classification)"
    
    # 解析分类报告
    lines = report.strip().split('\n')
    
    # 提取每个类别的指标
    precision_values = []
    recall_values = []
    f1_values = []
    support_values = []
    
    # 查找表格数据行
    for line in lines:
        line = line.strip()
        if line.startswith('Class '):
            # 解析类别行，例如: "Class 0     0.5000    0.5763    0.5354        59"
            parts = line.split()
            if len(parts) >= 5:
                precision_values.append(float(parts[2]))
                recall_values.append(float(parts[3]))
                f1_values.append(float(parts[4]))
                support_values.append(int(parts[5]))
    
    # 添加详细指标 - 按类别分组
    class_metrics = {}
    for i in range(len(precision_values)):
        class_metrics[f"class_{i}"] = {
            "precision": precision_values[i],
            "recall": recall_values[i],
            "f1-score": f1_values[i],
            "support": support_values[i]
        }
    
    json_data["class_metrics"] = class_metrics
    
    # 提取 accuracy, macro avg, weighted avg
    for line in lines:
        line = line.strip()
        if line.startswith('accuracy'):
            # 例如: "accuracy                         0.4891       640"
            parts = line.split()
            if len(parts) >= 2:
                json_data["accuracy"] = float(parts[1])
        elif line.startswith('macro avg'):
            # 例如: "macro avg     0.4820    0.4911    0.4831       640"
            parts = line.split()
            if len(parts) >= 5:
                json_data["macro_avg_precision"] = float(parts[2])
                json_data["macro_avg_recall"] = float(parts[3])
                json_data["macro_avg_f1"] = float(parts[4])
                json_data["macro_avg_support"] = int(parts[5])
        elif line.startswith('weighted avg'):
            # 例如: "weighted avg     0.4862    0.4891    0.4839       640"
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
    解析分类报告并生成 JSON 数据 (批量投票版本)
    """
    # 构建批量投票的基础元数据
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
    
    # 使用通用解析函数
    return parse_classification_report_to_json(report, base_metadata, auc_score)


def save_single_file_results(output_txt_file, input_jsonl_file, dataset_name, 
                           report, auc_score, total_samples, bad_sample_count):
    """
    保存单文件评估结果 (TXT + JSON)
    
    Args:
        output_txt_file: 输出的TXT文件路径
        input_jsonl_file: 输入的JSONL文件路径
        dataset_name: 数据集名称
        report: 分类报告字符串
        auc_score: AUC分数
        total_samples: 总样本数
        bad_sample_count: 错误样本数
    """
    # 构建单文件评估的基础元数据
    base_metadata = {
        "Dataset": dataset_name,
        "Input file": input_jsonl_file,
        "Total samples": total_samples,
        "Responses with wrong sample size": bad_sample_count,
        "Evaluation mode": "Direct single file assessment"
    }
    
    # 使用通用解析函数生成JSON数据
    json_data = parse_classification_report_to_json(report, base_metadata, auc_score)
    
    # 保存TXT文件
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        f.write(f"Single File Evaluation Report:\n\n")
        f.write(f"Input file: {input_jsonl_file}\n")
        f.write(f"Total samples processed: {total_samples}\n")
        f.write(f"Responses with wrong sample size: {bad_sample_count}\n\n")
        f.write(report)
        if auc_score is not None:
            f.write(f"\nAUC Score: {auc_score:.4f}\n")
    
    # 智能生成JSON文件路径
    if output_txt_file.endswith('.txt'):
        json_path = output_txt_file.replace('.txt', '.json')
    elif output_txt_file.endswith('.json'):
        # 如果用户指定的是.json文件，我们需要生成对应的.txt文件
        json_path = output_txt_file
        txt_path = output_txt_file.replace('.json', '.txt')
        # 重新保存TXT文件到正确的路径
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Single File Evaluation Report:\n\n")
            f.write(f"Input file: {input_jsonl_file}\n")
            f.write(f"Total samples processed: {total_samples}\n")
            f.write(f"Responses with wrong sample size: {bad_sample_count}\n\n")
            f.write(report)
            if auc_score is not None:
                f.write(f"\nAUC Score: {auc_score:.4f}\n")
        print(f"SUCCESS: Results saved to: {txt_path}")
    else:
        # 如果没有扩展名，默认添加.json
        json_path = output_txt_file + '.json'
    
    # 保存JSON文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
        
    if not output_txt_file.endswith('.json'):
        print(f"SUCCESS: Results saved to: {output_txt_file}")
    print(f"SUCCESS: JSON data saved to: {json_path}")


def save_results(result_output_dir, dataset_name, model_name, row_shuffle_seeds,
                train_chunk_size, test_chunk_size, weighted,
                report, auc_score, processed_combinations, bad_sample_count):
    """
    保存评估结果
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

    # 解析分类报告并生成 JSON 文件
    base_filename = f"{dataset_name}_{voting_type}_trainSize{train_chunk_size}_{config_str}"
    json_file = os.path.join(result_output_dir, f"{base_filename}.json")
    
    json_data = _parse_classification_report(report, dataset_name, model_name, voting_method, 
                                            row_shuffle_seeds, 
                                            len(processed_combinations), processed_combinations,
                                            train_chunk_size, test_chunk_size, 
                                            bad_sample_count, auc_score)
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"SUCCESS: Results saved to: {result_file}")
    print(f"SUCCESS: JSON data saved to: {json_file}")
