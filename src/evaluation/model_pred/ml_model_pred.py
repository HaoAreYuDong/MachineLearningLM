import os
import sys
import json
import argparse
import concurrent.futures
from threading import Lock
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from ml_utils import (
    BaseMLRunner, 
    determine_input_output_paths,
    extract_model_prefix,
    construct_file_paths
)


class MLModelRunner(BaseMLRunner):
    
    def __init__(self, model_name, max_workers=4, random_state=42):
        super().__init__(model_name, random_state)
        
        self.max_workers = max_workers
        self.lock = Lock()
        
        print(f"🤖 Initialized ML Model Runner: {model_name}")
        print(f"   Max Workers: {max_workers}")
        print(f"   Random State: {random_state}")

    def train_and_predict_single_with_lock(self, train_data, test_data, index, label_mapping):
        try:
            result = self.train_and_predict_single(train_data, test_data, index, label_mapping)
            
            with self.lock:
                print(f"   Index {index}: Accuracy = {result['accuracy']:.4f}, "
                      f"Train labels: {list(result['label_mapping']['unified_to_original'].values())}, "
                      f"Test labels: {list(result['label_mapping']['unified_to_original'].values())}")
            
            return result
            
        except Exception as e:
            import traceback
            with self.lock:
                print(f"❌ Error processing index {index}:")
                print(f"   Exception type: {type(e).__name__}")
                print(f"   Exception message: {str(e)}")
                print(f"   Detailed traceback:")
                traceback.print_exc()
                print(f"   Train data paths: {train_data}")
                print(f"   Test data paths: {test_data}")
            return None

    def process_dataset(self, input_dir, output_file, dataset_name, split_seed, 
                       row_shuffle_seed, train_chunk_size, test_chunk_size, 
                       max_samples=None, force_overwrite=False):
        
        subdir = f"{dataset_name}_Sseed{split_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}"
        rseed_dir = f"Rseed{row_shuffle_seed}"
        
        split_dir = os.path.join(input_dir, dataset_name, subdir)
        data_dir = os.path.join(split_dir, rseed_dir)
        
        if not os.path.exists(data_dir):
            print(f"❌ Input directory not found: {data_dir}")
            return False
        
        if os.path.exists(output_file) and not force_overwrite:
            print(f"⚠️  Output file already exists: {output_file}")
            response = input("Do you want to overwrite it? (y/n): ")
            if response.lower() != 'y':
                print("Skipping...")
                return True
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        label_mapping = self.load_label_mapping(split_dir)
        print(f"📊 Loaded label mapping: {label_mapping}")
        
        X_train_dir = os.path.join(data_dir, "X_train")
        X_test_dir = os.path.join(data_dir, "X_test")
        y_train_dir = os.path.join(data_dir, "y_train")
        y_test_dir = os.path.join(data_dir, "y_test")
        
        for dir_path in [X_train_dir, X_test_dir, y_train_dir, y_test_dir]:
            if not os.path.exists(dir_path):
                print(f"❌ Required directory not found: {dir_path}")
                return False
        
        csv_files = sorted([f for f in os.listdir(X_train_dir) if f.endswith('.csv')])
        if max_samples:
            csv_files = csv_files[:max_samples]
        
        print(f"📊 Found {len(csv_files)} CSV files to process")
        
        tasks = []
        for csv_file in csv_files:
            index = int(csv_file.split('.')[0])
            
            train_data = {
                'X_train': os.path.join(X_train_dir, csv_file),
                'y_train': os.path.join(y_train_dir, csv_file)
            }
            test_data = {
                'X_test': os.path.join(X_test_dir, csv_file),
                'y_test': os.path.join(y_test_dir, csv_file)
            }
            
            tasks.append((train_data, test_data, index, label_mapping))
        
        results = []
        print(f"🔄 Processing {len(tasks)} tasks with {self.max_workers} workers...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.train_and_predict_single_with_lock, *task): task[2] 
                for task in tasks
            }
            
            with tqdm(total=len(tasks), desc="ML Training & Prediction") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as exc:
                        print(f"⚠️  Task {index} generated an exception: {exc}")
                    pbar.update(1)
        
        results.sort(key=lambda x: x['id'])
        
        with open(output_file, 'w') as f:
            for result in results:
                output_result = {
                    "id": result["id"],
                    "response": result["response"], 
                    "groundtruth": result["groundtruth"],
                    "batch_probabilities": result["batch_probabilities"],
                    "available_labels": result["available_labels"]
                }
                f.write(json.dumps(output_result) + '\n')
        
        if results:
            avg_accuracy = np.mean([r['accuracy'] for r in results])
            print(f"✅ Processing completed! Average accuracy: {avg_accuracy:.4f}")
            print(f"📁 Output saved to: {output_file}")
        else:
            print(f"❌ No results generated")
            return False
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Machine Learning Model Batch Training and Prediction Script')
    
    # Input/output parameters
    parser.add_argument("--input_dir", required=True,
                        help="Input directory path (1_split directory)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory path (3_predict directory)")
    
    # Batch processing parameters
    parser.add_argument("--dataset_names", required=True,
                        help="Dataset name list, space-separated, e.g.: 'bank heloc rl'")
    parser.add_argument("--row_shuffle_seeds", required=True,
                        help="Row shuffle seed list, space-separated, e.g.: '40 41 42'")

    # Path construction parameters
    parser.add_argument("--split_seed", type=int, default=42,
                       help="Split seed (default: 42)")
    parser.add_argument("--train_chunk_size", type=int, default=8,
                        help="Training chunk size (default: 8)")
    parser.add_argument("--test_chunk_size", type=int, default=50,
                        help="Test chunk size (default: 50)")

    # Model parameters
    parser.add_argument('--model_name', type=str, required=True, 
                        help='Machine learning model name: randomforest/rf, xgboost/xgb, knn')
    
    # Optional parameters
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of samples (number of CSV files)')
    parser.add_argument('--force_overwrite', action='store_true', default=False,
                        help='If set, will directly delete existing output files without prompting')
    
    # Parallelization parameters
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads for parallel processing (default: 4)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if not args.input_dir or not args.output_dir:
        print("❌ Error: Both --input_dir and --output_dir are required")
        sys.exit(1)
    
    dataset_names = args.dataset_names.split() if args.dataset_names else []
    row_shuffle_seeds = [int(x) for x in args.row_shuffle_seeds.split()] if args.row_shuffle_seeds else []
    
    if not dataset_names or not row_shuffle_seeds:
        print("❌ Error: Both --dataset_names and --row_shuffle_seeds are required")
        sys.exit(1)
    
    print(f"🚀 Initializing ML model runner...")
    try:
        runner = MLModelRunner(args.model_name, args.max_workers, args.random_state)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    total_tasks = len(dataset_names) * len(row_shuffle_seeds)
    current_task = 0
    successful_tasks = 0
    failed_tasks = 0
    
    print(f"📊 Batch processing summary:")
    print(f"   Datasets: {dataset_names}")
    print(f"   Row shuffle seeds: {row_shuffle_seeds}")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Model: {args.model_name}")
    print("")
    
    try:
        for dataset_name in dataset_names:
            for row_shuffle_seed in row_shuffle_seeds:
                current_task += 1
                
                print(f"┌─────────────────────────────────────────────────────────────────────────────┐")
                print(f"│ Task {current_task:2d}/{total_tasks:2d} │ Dataset: {dataset_name:10s} │ Seed: {row_shuffle_seed:4d} │")
                print(f"└─────────────────────────────────────────────────────────────────────────────┘")
                
                try:
                    input_path, output_file = determine_input_output_paths(
                        args.input_dir, args.output_dir, dataset_name,
                        args.split_seed, row_shuffle_seed, args.train_chunk_size,
                        args.test_chunk_size, args.model_name
                    )
                    
                    print(f"📂 Paths:")
                    print(f"   Input:  {input_path}")
                    print(f"   Output: {output_file}")
                    
                    success = runner.process_dataset(
                        args.input_dir, output_file, dataset_name,
                        args.split_seed, row_shuffle_seed, args.train_chunk_size,
                        args.test_chunk_size, args.max_samples, args.force_overwrite
                    )
                    
                    if success:
                        print(f"✅ Successfully processed: {dataset_name} (seed: {row_shuffle_seed})")
                        successful_tasks += 1
                    else:
                        print(f"❌ Failed to process: {dataset_name} (seed: {row_shuffle_seed})")
                        failed_tasks += 1
                        
                except (FileNotFoundError, ValueError) as e:
                    print(f"❌ Error processing {dataset_name} (seed: {row_shuffle_seed}): {e}")
                    failed_tasks += 1
                
                print("")
        
        print("🎉 Batch processing completed!")
        print(f"📊 Final statistics:")
        print(f"   Total tasks: {total_tasks}")
        print(f"   Successful: {successful_tasks}")
        print(f"   Failed: {failed_tasks}")
        print(f"   Success rate: {successful_tasks/total_tasks*100:.1f}%")
        
        if failed_tasks > 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
