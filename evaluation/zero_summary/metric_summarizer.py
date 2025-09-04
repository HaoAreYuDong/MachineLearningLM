#!/usr/bin/env python3
"""
简化版 Metric Results Summarizer
"""

import os
import sys
import json
import pandas as pd
import glob
import re
from datetime import datetime
import pytz


def get_beijing_time():
    """获取北京时间格式化字符串（月日时分）"""
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(beijing_tz)
    return now.strftime("%m%d%H%M")


def extract_model_prefix(model_name):
    """提取模型名称前缀"""
    # 处理 backend::model 格式
    if '::' in model_name:
        parts = model_name.split('::', 1)
        backend, actual_model = parts[0], parts[1]
        if backend.lower() == 'openai':
            # 对于 openai::model，使用后面的模型名
            model_name = actual_model
        else:
            # 对于其他 backend，使用完整的格式
            model_name = model_name.replace('::', '_')
    
    # 处理 HuggingFace 格式的路径（如 minzl/toy_3550）
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    
    # 替换所有可能的特殊字符
    return model_name.replace('-', '_').replace('.', '_').replace(':', '@').replace('/', '_')


def parse_json_file(json_file_path):
    """解析JSON文件"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 从Input file解析信息
        input_file = data.get('Input file', '')
        filename = os.path.basename(input_file)
        
        # 提取模型名称
        model_name = 'unknown'
        if '@@' in filename:
            model_name = filename.split('@@')[0]
        
        # 提取训练和测试大小 - 适配新的文件命名格式
        train_size = None
        test_size = None
        # 新格式: model@@dataset_Sseed*_trainsize*_testsize*_seed@*_report
        train_match = re.search(r'trainsize(\d+)', input_file)
        test_match = re.search(r'testsize(\d+)', input_file)
        if train_match:
            train_size = int(train_match.group(1))
        if test_match:
            test_size = int(test_match.group(1))
        
        # 如果新格式没有匹配到，尝试旧格式兼容
        if train_size is None:
            train_match_old = re.search(r'trainSize(\d+)', input_file)
            if train_match_old:
                train_size = int(train_match_old.group(1))
        
        if test_size is None:
            test_match_old = re.search(r'testSize(\d+)', input_file)
            if test_match_old:
                test_size = int(test_match_old.group(1))
        
        # 读取accuracy（从txt文件）
        accuracy = None
        txt_file_path = json_file_path.replace('.json', '.txt')
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            accuracy_match = re.search(r'accuracy\s+(\d+\.\d+)', content)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
        
        # 获取类别指标
        class_metrics = data.get('class_metrics', {})
        f1_c0 = class_metrics.get('class_0', {}).get('f1-score', None)
        f1_c1 = class_metrics.get('class_1', {}).get('f1-score', None)
        
        # 处理AUC
        auc_score = data.get('AUC Score', None)
        if auc_score is None or auc_score == "Not calculated (requires binary classification)":
            auc_score = -1
        
        return {
            'dataset': data.get('Dataset', 'unknown'),
            'train_size': train_size,
            'test_size': test_size,
            'model_name': model_name,
            'total_samples': data.get('Total samples', 0),
            'error_samples': data.get('Responses with wrong sample size', 0),
            'f1_c0': f1_c0,
            'f1_c1': f1_c1,
            'f1_w': data.get('weighted_avg_f1', None),
            'auc': auc_score,
            'accuracy': accuracy
        }
        
    except Exception as e:
        print(f"Error parsing {json_file_path}: {e}")
        return None


def main():
    
    
    if len(sys.argv) < 5:
        print("Usage: python script.py --metric_data_dir DIR --report_data_dir DIR [--model_name NAME]")
        return 1
    
    # 简单解析参数
    metric_data_dir = None
    report_data_dir = None
    model_name_hint = None
    
    for i, arg in enumerate(sys.argv):
        if arg == '--metric_data_dir' and i + 1 < len(sys.argv):
            metric_data_dir = sys.argv[i + 1]
        elif arg == '--report_data_dir' and i + 1 < len(sys.argv):
            report_data_dir = sys.argv[i + 1]
        elif arg == '--model_name' and i + 1 < len(sys.argv):
            model_name_hint = sys.argv[i + 1]
    
    if not metric_data_dir or not report_data_dir:
        print("Both --metric_data_dir and --report_data_dir are required")
        return 1
    
    print(f"STARTING: 开始统计metric数据...")
    print(f"INPUT: 输入目录: {metric_data_dir}")
    print(f"OUTPUT: 输出目录: {report_data_dir}")
    if model_name_hint:
        print(f"🏷️  模型名称: {model_name_hint}")
    print()
    
    # 查找JSON文件
    json_files = glob.glob(os.path.join(metric_data_dir, "**", "*.json"), recursive=True)
    print(f"OUTPUT: 找到 {len(json_files)} 个JSON文件")
    
    if not json_files:
        print("ERROR: 没有找到任何JSON文件")
        return 1
    
    # 解析所有文件
    results = []
    for json_file in json_files:
        result = parse_json_file(json_file)
        if result:
            results.append(result)
    
    print(f"SUCCESS: 成功解析 {len(results)}/{len(json_files)} 个文件")
    
    if not results:
        print("ERROR: 没有成功解析任何文件")
        return 1
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 按dataset排序，然后按train_size排序
    df_sorted = df.sort_values(['dataset', 'train_size'], ascending=[True, True])
    
    # 确定模型前缀
    if model_name_hint:
        model_prefix = extract_model_prefix(model_name_hint)
    else:
        # 使用最常见的模型名
        model_names = [r['model_name'] for r in results if r['model_name'] != 'unknown']
        if model_names:
            model_prefix = extract_model_prefix(model_names[0])
        else:
            model_prefix = 'unknown'
    
    # 生成文件名
    beijing_time = get_beijing_time()
    csv_filename = f"{model_prefix}_{beijing_time}_results.csv"
    csv_path = os.path.join(report_data_dir, csv_filename)
    
    # 确保输出目录存在
    os.makedirs(report_data_dir, exist_ok=True)
    
    # 按指定顺序排列列
    column_order = [
        'dataset', 'train_size', 'test_size', 'model_name', 'total_samples',
        'error_samples', 'f1_c0', 'f1_c1', 'f1_w', 'auc', 'accuracy'
    ]
    
    df_ordered = df_sorted.reindex(columns=column_order)
    
    # 美化CSV输出：在数据集之间插入空行
    def write_formatted_csv(df, file_path):
        """写入格式化的CSV文件，在数据集之间插入空行"""
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            # 写入表头
            f.write(','.join(df.columns) + '\n')
            
            current_dataset = None
            for _, row in df.iterrows():
                # 如果是新的数据集且不是第一行，插入空行
                if current_dataset is not None and row['dataset'] != current_dataset:
                    f.write('\n')
                
                # 写入数据行
                row_values = []
                for col in df.columns:
                    value = row[col]
                    # 格式化数值，保持精度
                    if isinstance(value, float) and not pd.isna(value):
                        if col in ['f1_c0', 'f1_c1', 'f1_w', 'accuracy']:
                            row_values.append(f"{value:.4f}")
                        elif col == 'auc':
                            row_values.append(f"{value:.6f}")
                        else:
                            row_values.append(str(value))
                    else:
                        row_values.append(str(value) if not pd.isna(value) else '')
                
                f.write(','.join(row_values) + '\n')
                current_dataset = row['dataset']
    
    # 使用自定义函数写入CSV
    write_formatted_csv(df_ordered, csv_path)
    
    # 统计信息
    datasets_count = df_ordered['dataset'].nunique()
    train_sizes = sorted(df_ordered['train_size'].dropna().unique())
    
    print(f"INFO: CSV文件已生成: {csv_path}")
    print(f"📈 统计信息:")
    print(f"   - 数据集数量: {datasets_count}")
    print(f"   - 训练大小: {train_sizes}")
    print(f"   - 总记录数: {len(df_ordered)}")
    print(f"   - 排序方式: 数据集名称 → 训练大小")
    print(f"   - 格式化: 数据集间插入空行")
    return 0


if __name__ == "__main__":
    exit(main())
