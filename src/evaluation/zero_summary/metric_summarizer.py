#!/usr/bin/env python3
"""
Simplified Metric Results Summarizer
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
    """Get formatted Beijing time string (month, day, hour, minute)"""
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(beijing_tz)
    return now.strftime("%m%d%H%M")


def extract_model_prefix(model_name):
    """Extract model name prefix"""
    # Handle backend::model format
    if '::' in model_name:
        parts = model_name.split('::', 1)
        backend, actual_model = parts[0], parts[1]
        if backend.lower() == 'openai':
            # For openai::model, use the actual model name
            model_name = actual_model
        else:
            # For other backends, use full format
            model_name = model_name.replace('::', '_')
    
    # Handle HuggingFace format paths (e.g., minzl/toy_3550)
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    
    # Replace all possible special characters
    return model_name.replace('-', '_').replace('.', '_').replace(':', '@').replace('/', '_')


def parse_json_file(json_file_path):
    """Parse JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract information from Input file
        input_file = data.get('Input file', '')
        filename = os.path.basename(input_file)
        
        # Extract model name
        model_name = 'unknown'
        if '@@' in filename:
            model_name = filename.split('@@')[0]
        
        # Extract training and test sizes - adapt to new file naming format
        train_size = None
        test_size = None
        # New format: model@@dataset_Sseed*_trainsize*_testsize*_seed@*_report
        train_match = re.search(r'trainsize(\d+)', input_file)
        test_match = re.search(r'testsize(\d+)', input_file)
        if train_match:
            train_size = int(train_match.group(1))
        if test_match:
            test_size = int(test_match.group(1))
        
        # If new format not matched, try old format compatibility
        if train_size is None:
            train_match_old = re.search(r'trainSize(\d+)', input_file)
            if train_match_old:
                train_size = int(train_match_old.group(1))
        
        if test_size is None:
            test_match_old = re.search(r'testSize(\d+)', input_file)
            if test_match_old:
                test_size = int(test_match_old.group(1))
        
        # Read accuracy (from txt file)
        accuracy = None
        txt_file_path = json_file_path.replace('.json', '.txt')
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            accuracy_match = re.search(r'accuracy\s+(\d+\.\d+)', content)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
        
        # Get class metrics
        class_metrics = data.get('class_metrics', {})
        f1_c0 = class_metrics.get('class_0', {}).get('f1-score', None)
        f1_c1 = class_metrics.get('class_1', {}).get('f1-score', None)
        
        # Handle AUC
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
    
    # Simple argument parsing
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
    
    print(f"🚀 Starting metric data collection...")
    print(f"📂 Input directory: {metric_data_dir}")
    print(f"📁 Output directory: {report_data_dir}")
    if model_name_hint:
        print(f"🏷️  Model name: {model_name_hint}")
    print()
    
    # Find JSON files
    json_files = glob.glob(os.path.join(metric_data_dir, "**", "*.json"), recursive=True)
    print(f"📁 Found {len(json_files)} JSON files")
    
    if not json_files:
        print("❌ No JSON files found")
        return 1
    
    # Parse all files
    results = []
    for json_file in json_files:
        result = parse_json_file(json_file)
        if result:
            results.append(result)
    
    print(f"✅ Successfully parsed {len(results)}/{len(json_files)} files")
    
    if not results:
        print("❌ No files parsed successfully")
        return 1
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by dataset, then by train_size
    df_sorted = df.sort_values(['dataset', 'train_size'], ascending=[True, True])
    
    # Determine model prefix
    if model_name_hint:
        model_prefix = extract_model_prefix(model_name_hint)
    else:
        # Use most common model name
        model_names = [r['model_name'] for r in results if r['model_name'] != 'unknown']
        if model_names:
            model_prefix = extract_model_prefix(model_names[0])
        else:
            model_prefix = 'unknown'
    
    # Generate filename
    beijing_time = get_beijing_time()
    csv_filename = f"{model_prefix}_{beijing_time}_results.csv"
    csv_path = os.path.join(report_data_dir, csv_filename)
    
    # Ensure output directory exists
    os.makedirs(report_data_dir, exist_ok=True)
    
    # Order columns
    column_order = [
        'dataset', 'train_size', 'test_size', 'model_name', 'total_samples',
        'error_samples', 'f1_c0', 'f1_c1', 'f1_w', 'auc', 'accuracy'
    ]
    
    df_ordered = df_sorted.reindex(columns=column_order)
    
    # Format CSV output: insert blank lines between datasets
    def write_formatted_csv(df, file_path):
        """Write formatted CSV with blank lines between datasets"""
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            # Write header
            f.write(','.join(df.columns) + '\n')
            
            current_dataset = None
            for _, row in df.iterrows():
                # If new dataset and not first row, insert blank line
                if current_dataset is not None and row['dataset'] != current_dataset:
                    f.write('\n')
                
                # Write data row
                row_values = []
                for col in df.columns:
                    value = row[col]
                    # Format numeric values, preserve precision
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
    
    # Write CSV using custom function
    write_formatted_csv(df_ordered, csv_path)
    
    # Statistics
    datasets_count = df_ordered['dataset'].nunique()
    train_sizes = sorted(df_ordered['train_size'].dropna().unique())
    
    print(f"📊 CSV file generated: {csv_path}")
    print(f"📈 Statistics:")
    print(f"   - Number of datasets: {datasets_count}")
    print(f"   - Training sizes: {train_sizes}")
    print(f"   - Total records: {len(df_ordered)}")
    print(f"   - Sorting: by dataset name → training size")
    print(f"   - Formatting: blank lines between datasets")
    return 0


if __name__ == "__main__":
    exit(main())