import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from tabicl import TabICLClassifier 
import time
import argparse
import traceback


DEFAULT_SAMPLE_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]

def save_split_data(base_path, dataset_name, X_sampled, X_test, y_sampled, y_test, sample_size):
    dataset_dir = os.path.join(base_path, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    sample_dir = os.path.join(dataset_dir, f"sample_{sample_size}")
    os.makedirs(sample_dir, exist_ok=True)
    
    X_sampled.to_csv(os.path.join(sample_dir, "X_sampled.csv"), index=False)
    
    pd.DataFrame(y_sampled).to_csv(
        os.path.join(sample_dir, "y_sampled.csv"), 
        index=False, 
        header=True
    )
    
    X_test.to_csv(os.path.join(sample_dir, "X_test.csv"), index=False)
    
    pd.DataFrame(y_test).to_csv(
        os.path.join(sample_dir, "y_test.csv"), 
        index=False, 
        header=True
    )

def calculate_auc(y_test, y_proba, n_classes, class_order=None):
    if y_proba is None or len(y_proba) == 0 or n_classes < 2:
        return np.nan
    
    le = LabelEncoder()
    y_test_numeric = le.fit_transform(y_test)
    unique_classes = le.classes_
    
    if n_classes == 2:
        if y_proba.shape[1] == 1:
            return roc_auc_score(y_test_numeric, y_proba[:, 0])
        
        elif y_proba.shape[1] == 2:
            if class_order is not None:
                positive_idx = 1 if unique_classes[1] in class_order else 0
            else:
                positive_idx = 1
            
            return roc_auc_score(y_test_numeric, y_proba[:, positive_idx])
        
        else:
            print(f"  Unexpected probability shape {y_proba.shape} for binary task")
            return np.nan
    
    else:
        if y_proba.shape[1] != n_classes:
            corrected_proba = np.zeros((len(y_test), n_classes))
            
            if class_order is not None and len(class_order) == y_proba.shape[1]:
                class_order = np.array(class_order)
                
                model_to_global = {}
                for model_idx, model_class in enumerate(class_order):
                    if model_class in unique_classes:
                        global_idx = np.where(unique_classes == model_class)[0][0]
                        model_to_global[model_idx] = global_idx
                
                for model_idx, global_idx in model_to_global.items():
                    corrected_proba[:, global_idx] = y_proba[:, model_idx]
                return roc_auc_score(y_test_numeric, corrected_proba, multi_class='ovr', average='weighted')
            
            else:
                num_common = min(n_classes, y_proba.shape[1])
                corrected_proba[:, :num_common] = y_proba[:, :num_common]
                return roc_auc_score(y_test_numeric, corrected_proba, multi_class='ovr', average='weighted')
        
        return roc_auc_score(
            y_test_numeric, y_proba, 
            multi_class='ovr', 
            average='weighted'
        )

def process_dataset(file_path, output_dir, sample_sizes):
    dataset_name = os.path.basename(file_path).split('.')[0]
    print(f"Processing dataset: {dataset_name}")
    
    split_data_path = os.path.join(output_dir, "split_data")
    error_log_path = os.path.join(output_dir, "error_logs")
    os.makedirs(split_data_path, exist_ok=True)
    os.makedirs(error_log_path, exist_ok=True)
    
    start_time = time.time()
    
    try:
        df = pd.read_csv(file_path)
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = []
        n_classes = len(np.unique(y))
        print(f"  Task type: {'Binary' if n_classes == 2 else 'Multi-class'} classification with {n_classes} classes")
        
        for sample_size in sample_sizes:
            sample_start = time.time()
            
            actual_size = min(sample_size, len(X_train))
            print(f"  Processing sample size: {actual_size}")
            
            if actual_size < len(X_train):
                unique_classes = []
                attempts = 0
                max_attempts = 10 
                
                while len(unique_classes) <= 1 and attempts < max_attempts:
                    sampled_idx = np.random.RandomState(seed=42+attempts).choice(
                        X_train.index, actual_size, replace=False
                    )
                    y_sampled = y_train.loc[sampled_idx]
                    unique_classes = np.unique(y_sampled)
                    attempts += 1
                
                if len(unique_classes) <= 1:
                    print(f"  Warning: Only {len(unique_classes)} class after {attempts} attempts")
                
                X_sampled = X_train.loc[sampled_idx]
                y_sampled = y_train.loc[sampled_idx]
            else:
                X_sampled, y_sampled = X_train, y_train
            
            save_split_data(split_data_path, dataset_name, X_sampled, X_test, y_sampled, y_test, sample_size)
            
            model = TabICLClassifier(
                n_estimators=32,                                        # number of ensemble members
                norm_methods=["none", "power"],                         # normalization methods to try
                feat_shuffle_method="latin",                            # feature permutation strategy
                class_shift=True,                                       # whether to apply cyclic shifts to class labels
                outlier_threshold=4.0,                                  # z-score threshold for outlier detection and clipping
                softmax_temperature=0.9,                                # controls prediction confidence
                average_logits=True,                                    # whether ensemble averaging is done on logits or probabilities
                use_hierarchical=True,                                  # enable hierarchical classification for datasets with many classe
                batch_size=8,                                           # process this many ensemble members together (reduce RAM usage)
                use_amp=True,                                           # use automatic mixed precision for faster inference
                model_path=None,                                        # where the model checkpoint is stored
                allow_auto_download=True,                               # whether automatic download to the specified path is allowed
                checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",  # the version of pretrained checkpoint to use
                device=None,                                            # specify device for inference
                random_state=42,                                        # random seed for reproducibility
                n_jobs=None,                                            # number of threads to use for PyTorch
                verbose=False,                                          # print detailed information during inference
                inference_config=None,                                  # inference configuration for fine-grained control
             )
            model.fit(X_sampled, y_sampled)
            
            y_pred = model.predict(X_test)
            
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                else:
                    y_proba = None
            except Exception as e:
                print(f"  Error in predict_proba: {str(e)}")
                y_proba = None
            
            acc = accuracy_score(y_test, y_pred)
            weighted_f1 = f1_score(y_test, y_pred, average='weighted')
            f1_scores = f1_score(y_test, y_pred, average=None)
            
            
            class_order = None
            if hasattr(model, 'classes_'):
                class_order = model.classes_

            auc = calculate_auc(y_test, y_proba, n_classes, class_order=class_order)
            
            result_row = {
                'dataset': dataset_name,
                'model': 'tabicl',
                'len(train_set)': actual_size,
                'acc': acc,
                'weighted_f1': weighted_f1,
                'auc': auc
            }
            
            for i, score in enumerate(f1_scores):
                result_row[f'f1_class_{i}'] = score
            
            results.append(result_row)
            
            sample_duration = time.time() - sample_start
            print(f"  Sample size {actual_size} processed in {sample_duration:.2f}s")
        
        total_duration = time.time() - start_time
        print(f"Dataset {dataset_name} completed in {total_duration:.2f}s")
        return results
    
    except Exception as e:
        error_msg = f"Error processing dataset {dataset_name}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        os.makedirs(error_log_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        error_filename = os.path.join(error_log_path, f"{dataset_name}_error_{timestamp}.txt")
        
        with open(error_filename, 'w') as f:
            f.write(error_msg)
        
        return {'error': error_msg, 'dataset': dataset_name}

def main():
    parser = argparse.ArgumentParser(description='Evaluate the performance of the TabICL model on multiple datasets.')
    parser.add_argument('--datasets', nargs='+', default=["all"],
                        help='Name of the dataset(s) to process (multiple names should be space-separated), or use "all" to process all datasets. Default: all')
    parser.add_argument('--data_dir', default="./datahub_inputs/data_raw",
                        help='Path to the directory containing raw datasets. Default: ./datahub_inputs/data_raw')
    parser.add_argument('--output_dir', default="./datahub_outputs/tabicl",
                        help='Path to the output directory. Default: ./datahub_outputs/tabicl')
    parser.add_argument('--n_jobs', type=int, default=2,
                        help='Number of parallel jobs to run. Default: 2')
    parser.add_argument('--sample_sizes', nargs='+', type=int, default=DEFAULT_SAMPLE_SIZES,
                        help='List of sample sizes to evaluate. Default: 8 16 32 64 128 256 512 1024')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found at {args.data_dir}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_files = [
        os.path.join(args.data_dir, f) 
        for f in os.listdir(args.data_dir) 
        if f.endswith('.csv')
    ]
    
    if "all" in args.datasets:
        selected_files = all_files
        print("Processing all datasets")
    else:
        selected_files = []
        for dataset in args.datasets:
            file_path = os.path.join(args.data_dir, f"{dataset}.csv")
            if os.path.exists(file_path):
                selected_files.append(file_path)
            else:
                print(f"Warning: Dataset {dataset} not found at {file_path}")
        
        if not selected_files:
            print("No valid datasets specified. Exiting.")
            sys.exit(1)
    
    print(f"Found {len(selected_files)} datasets to process")
    print(f"Sample sizes: {args.sample_sizes}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parallel jobs: {args.n_jobs}")
    
    all_outputs = Parallel(n_jobs=args.n_jobs)(
        delayed(process_dataset)(file_path, args.output_dir, args.sample_sizes) 
        for file_path in selected_files
    )
    
    all_results = []
    errors = []
    
    for output in all_outputs:
        if isinstance(output, list):
            all_results.extend(output)
        elif isinstance(output, dict) and 'error' in output:
            errors.append(output)
    
    if errors:
        print(f"\nEncountered errors in {len(errors)} datasets")
        for error in errors:
            print(f"- {error['dataset']}")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        error_filename = os.path.join(args.output_dir, "error_logs", f"summary_errors_{timestamp}.txt")
        
        with open(error_filename, 'w') as f:
            f.write(f"Error Summary - {timestamp}\n")
            f.write("="*50 + "\n\n")
            
            for i, error_info in enumerate(errors, 1):
                f.write(f"Error {i}: Dataset '{error_info['dataset']}'\n")
                f.write("-"*50 + "\n")
                f.write(error_info['error'])
                f.write("\n\n" + "="*50 + "\n\n")
        
        print(f"Saved error summary to {error_filename}")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(
            by=['dataset', 'len(train_set)'], 
            ascending=[True, True]
        )
        
        column_order = ['dataset', 'model', 'len(train_set)', 'acc', 'weighted_f1']
        f1_columns = [col for col in results_df.columns if col.startswith('f1_class_')]
        column_order += sorted(f1_columns, key=lambda x: int(x.split('_')[-1]))
        column_order += ['auc']
        results_df = results_df[column_order]
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.csv")
        xlsx_filename = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.xlsx")
        
        results_df.to_csv(csv_filename, index=False)
        results_df.to_excel(xlsx_filename, index=False)
        print(f"\nEvaluation complete! Results saved to:")
        print(f"- CSV: {csv_filename}")
        print(f"- Excel: {xlsx_filename}")
    else:
        print("\nNo datasets were successfully processed.")

if __name__ == "__main__":
    main()