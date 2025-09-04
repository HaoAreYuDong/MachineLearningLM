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

# 导入我们的基础工具类
from ml_utils import (
    BaseMLRunner, 
    determine_input_output_paths,
    extract_model_prefix,
    construct_file_paths
)


class MLModelRunner(BaseMLRunner):
    """机器学习模型推理器，继承自BaseMLRunner"""
    
    def __init__(self, model_name, max_workers=4, random_state=42):
        # 调用父类初始化
        super().__init__(model_name, random_state)
        
        self.max_workers = max_workers
        self.lock = Lock()
        
        print(f"🤖 Initialized ML Model Runner: {model_name}")
        print(f"   Max Workers: {max_workers}")
        print(f"   Random State: {random_state}")

    def train_and_predict_single_with_lock(self, train_data, test_data, index, label_mapping):
        """带线程锁的训练和预测方法"""
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
                print(f"ERROR: Error processing index {index}:")
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
        """处理单个数据集"""
        
        # 构建输入路径
        subdir = f"{dataset_name}_Sseed{split_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}"
        rseed_dir = f"Rseed{row_shuffle_seed}"
        
        split_dir = os.path.join(input_dir, dataset_name, subdir)
        data_dir = os.path.join(split_dir, rseed_dir)
        
        # 检查输入目录是否存在
        if not os.path.exists(data_dir):
            print(f"ERROR: Input directory not found: {data_dir}")
            return False
        
        # 检查输出文件是否已存在
        if os.path.exists(output_file) and not force_overwrite:
            print(f"WARNING:  Output file already exists: {output_file}")
            response = input("Do you want to overwrite it? (y/n): ")
            if response.lower() != 'y':
                print("Skipping...")
                return True
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 加载标签映射
        label_mapping = self.load_label_mapping(split_dir)
        print(f"INFO: Loaded label mapping: {label_mapping}")
        
        # 获取所有数据文件
        X_train_dir = os.path.join(data_dir, "X_train")
        X_test_dir = os.path.join(data_dir, "X_test")
        y_train_dir = os.path.join(data_dir, "y_train")
        y_test_dir = os.path.join(data_dir, "y_test")
        
        # 检查必要的目录是否存在
        for dir_path in [X_train_dir, X_test_dir, y_train_dir, y_test_dir]:
            if not os.path.exists(dir_path):
                print(f"ERROR: Required directory not found: {dir_path}")
                return False
        
        # 获取CSV文件列表
        csv_files = sorted([f for f in os.listdir(X_train_dir) if f.endswith('.csv')])
        if max_samples:
            csv_files = csv_files[:max_samples]
        
        print(f"INFO: Found {len(csv_files)} CSV files to process")
        
        # 准备数据
        tasks = []
        for csv_file in csv_files:
            index = int(csv_file.split('.')[0])  # 从文件名提取索引
            
            train_data = {
                'X_train': os.path.join(X_train_dir, csv_file),
                'y_train': os.path.join(y_train_dir, csv_file)
            }
            test_data = {
                'X_test': os.path.join(X_test_dir, csv_file),
                'y_test': os.path.join(y_test_dir, csv_file)
            }
            
            tasks.append((train_data, test_data, index, label_mapping))
        
        # 并行处理
        results = []
        print(f"🔄 Processing {len(tasks)} tasks with {self.max_workers} workers...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self.train_and_predict_single_with_lock, *task): task[2] 
                for task in tasks
            }
            
            # 收集结果
            with tqdm(total=len(tasks), desc="ML Training & Prediction") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as exc:
                        print(f"WARNING:  Task {index} generated an exception: {exc}")
                    pbar.update(1)
        
        # 按索引排序结果
        results.sort(key=lambda x: x['id'])
        
        # 保存结果
        with open(output_file, 'w') as f:
            for result in results:
                # 移除内部使用的字段
                output_result = {
                    "id": result["id"],
                    "response": result["response"], 
                    "groundtruth": result["groundtruth"],
                    "batch_probabilities": result["batch_probabilities"],
                    "available_labels": result["available_labels"]
                }
                f.write(json.dumps(output_result) + '\n')
        
        # 计算平均准确率
        if results:
            avg_accuracy = np.mean([r['accuracy'] for r in results])
            print(f"SUCCESS: Processing completed! Average accuracy: {avg_accuracy:.4f}")
            print(f"OUTPUT: Output saved to: {output_file}")
        else:
            print(f"ERROR: No results generated")
            return False
        
        return True


def main():
    parser = argparse.ArgumentParser(description='机器学习模型批量训练和预测脚本')
    
    # 输入输出参数
    parser.add_argument("--input_dir", required=True,
                        help="输入目录路径（1_split目录）")
    parser.add_argument("--output_dir", required=True,
                        help="输出目录路径（3_predict目录）")
    
    # 批量处理参数
    parser.add_argument("--dataset_names", required=True,
                        help="数据集名称列表，用空格分隔，如：'bank heloc rl'")
    parser.add_argument("--row_shuffle_seeds", required=True,
                        help="行打乱种子列表，用空格分隔，如：'40 41 42'")

    # 路径构建参数
    parser.add_argument("--split_seed", type=int, default=42,
                       help="分割种子（默认：42）")
    parser.add_argument("--train_chunk_size", type=int, default=8,
                        help="训练块大小（默认：8）")
    parser.add_argument("--test_chunk_size", type=int, default=50,
                        help="测试块大小（默认：50）")

    # 模型参数
    parser.add_argument('--model_name', type=str, required=True, 
                        help='机器学习模型名称: randomforest/rf, xgboost/xgb, knn')
    
    # 可选参数
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='最大样本数（CSV文件数量）')
    parser.add_argument('--force_overwrite', action='store_true', default=False,
                        help='如果设置，将直接删除已存在的输出文件而不询问')
    
    # 并行化参数
    parser.add_argument('--max_workers', type=int, default=4,
                        help='并行处理的最大工作线程数（默认：4）')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子（默认：42）')
    
    args = parser.parse_args()
    
    # 验证必需参数
    if not args.input_dir or not args.output_dir:
        print("ERROR: Error: Both --input_dir and --output_dir are required")
        sys.exit(1)
    
    # 解析批量参数
    dataset_names = args.dataset_names.split() if args.dataset_names else []
    row_shuffle_seeds = [int(x) for x in args.row_shuffle_seeds.split()] if args.row_shuffle_seeds else []
    
    if not dataset_names or not row_shuffle_seeds:
        print("ERROR: Error: Both --dataset_names and --row_shuffle_seeds are required")
        sys.exit(1)
    
    # 初始化ML模型推理器
    print(f"STARTING: Initializing ML model runner...")
    try:
        runner = MLModelRunner(args.model_name, args.max_workers, args.random_state)
    except ValueError as e:
        print(f"ERROR: Error: {e}")
        sys.exit(1)
    
    # 计算总任务数
    total_tasks = len(dataset_names) * len(row_shuffle_seeds)
    current_task = 0
    successful_tasks = 0
    failed_tasks = 0
    
    print(f"INFO: Batch processing summary:")
    print(f"   Datasets: {dataset_names}")
    print(f"   Row shuffle seeds: {row_shuffle_seeds}")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Model: {args.model_name}")
    print("")
    
    try:
        # 双重循环处理所有组合
        for dataset_name in dataset_names:
            for row_shuffle_seed in row_shuffle_seeds:
                current_task += 1
                
                print(f"+-----------------------------------------------------------------------------+")
                print(f"| Task {current_task:2d}/{total_tasks:2d} | Dataset: {dataset_name:10s} | Seed: {row_shuffle_seed:4d} |")
                print(f"+-----------------------------------------------------------------------------+")
                
                try:
                    # 智能判断输入输出路径
                    input_path, output_file = determine_input_output_paths(
                        args.input_dir, args.output_dir, dataset_name,
                        args.split_seed, row_shuffle_seed, args.train_chunk_size,
                        args.test_chunk_size, args.model_name
                    )
                    
                    print(f"INPUT: Paths:")
                    print(f"   Input:  {input_path}")
                    print(f"   Output: {output_file}")
                    
                    # 处理数据集
                    success = runner.process_dataset(
                        args.input_dir, output_file, dataset_name,
                        args.split_seed, row_shuffle_seed, args.train_chunk_size,
                        args.test_chunk_size, args.max_samples, args.force_overwrite
                    )
                    
                    if success:
                        print(f"SUCCESS: Successfully processed: {dataset_name} (seed: {row_shuffle_seed})")
                        successful_tasks += 1
                    else:
                        print(f"ERROR: Failed to process: {dataset_name} (seed: {row_shuffle_seed})")
                        failed_tasks += 1
                        
                except (FileNotFoundError, ValueError) as e:
                    print(f"ERROR: Error processing {dataset_name} (seed: {row_shuffle_seed}): {e}")
                    failed_tasks += 1
                
                print("")
        
        # 打印最终统计
        print("COMPLETED: Batch processing completed!")
        print(f"INFO: Final statistics:")
        print(f"   Total tasks: {total_tasks}")
        print(f"   Successful: {successful_tasks}")
        print(f"   Failed: {failed_tasks}")
        print(f"   Success rate: {successful_tasks/total_tasks*100:.1f}%")
        
        if failed_tasks > 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nWARNING: Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
