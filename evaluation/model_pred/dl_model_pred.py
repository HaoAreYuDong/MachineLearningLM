import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import math
import argparse
from pathlib import Path
import sys
import re
import concurrent.futures
from threading import Lock

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

from tqdm import tqdm

# Import utility classes and functions
from dl_utils import BaseRunner, exp_safe

# All backend-specific packages are imported on-demand within their respective classes
# No longer importing transformers, openai, etc. at the top level


class OpenAIResult:
    """Unified OpenAI API result class"""
    
    def __init__(self, text, choice, logprobs_supported=True):
        self.text = text
        self.choice = choice
        self.logprobs_supported = logprobs_supported
        
        # Build outputs attribute for vLLM format compatibility
        self.outputs = [self]
        
        # Process logprobs information
        self.logprobs_data = None
        if logprobs_supported and choice and hasattr(choice, 'logprobs') and choice.logprobs:
            self.logprobs_data = self._process_logprobs(choice.logprobs)
    
    def _process_logprobs(self, logprobs):
        """Process OpenAI logprobs data and convert to unified format"""
        processed_logprobs = []
        
        if hasattr(logprobs, 'content') and logprobs.content:
            for token_logprob in logprobs.content:
                token_data = {
                    'token': token_logprob.token,
                    'logprob': token_logprob.logprob,
                    'top_logprobs': []
                }
                
                if hasattr(token_logprob, 'top_logprobs') and token_logprob.top_logprobs:
                    for top_choice in token_logprob.top_logprobs:
                        token_data['top_logprobs'].append({
                            'token': top_choice.token,
                            'logprob': top_choice.logprob
                        })
                
                processed_logprobs.append(token_data)
        
        return processed_logprobs


class VLLMRunner(BaseRunner):
    """vLLM inference engine"""
    
    def __init__(self, model_name, temperature=0.0):
        super().__init__(model_name, temperature)
        
        # Import vLLM and related packages on-demand, only when actually used
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"vLLM and transformers are required for local model inference. "
                f"Please install them with: pip install vllm transformers\n"
                f"Original error: {e}"
            )
        
        self.top_p = 0.7
        self.top_k = 50
        # top_logprobs is now unified in BaseRunner

        print(f"INFO: Initializing vLLM model: {model_name}")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.client = LLM(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=131072,
            rope_scaling={
                "type": "yarn",
                "rope_type": "yarn", 
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
            rope_theta=1_000_000.0,
            gpu_memory_utilization=0.89,
            swap_space=4,
            trust_remote_code=True,
        )
        
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            logprobs=self.top_logprobs,
            include_stop_str_in_output=False
        )
        
        print(f"SUCCESS: vLLM Model loaded successfully!")

    def generate(self, prompts):
        """Generate text using vLLM"""
        return self.client.generate(prompts, self.sampling_params)


class OpenAIRunner(BaseRunner):
    """OpenAI API inference engine (with parallelization support)"""
    
    def __init__(self, model_name, temperature=0.0, api_key=None, base_url=None, max_workers=4):
        super().__init__(model_name, temperature)

        # Import OpenAI package on-demand, only when actually used
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                f"openai package is required for OpenAI API inference. "
                f"Please install it with: pip install openai\n"
                f"Original error: {e}"
            )

        print(f"🌐 Initializing OpenAI API with model: {model_name}")
        self.temperature = temperature
        self.max_workers = max_workers

        # Use parameters first, then environment variables
        final_api_key = api_key or os.getenv('OPENAI_API_KEY')
        final_base_url = base_url or os.getenv('OPENAI_BASE_URL')
        
        if not final_api_key:
            raise ValueError("API key is required for OpenAI. Please provide --api_key or set OPENAI_API_KEY environment variable")
        if not final_base_url:
            raise ValueError("Base URL is required for OpenAI. Please provide --base_url or set OPENAI_BASE_URL environment variable")
        
        self.client = openai.OpenAI(api_key=final_api_key, base_url=final_base_url)
        self.logprobs_supported = True
        
        # Thread-safe lock
        self.lock = Lock()
        
        print(f"SUCCESS: OpenAI API initialized successfully!")
        print(f"   API Key: {final_api_key[:8]}...")
        print(f"   Base URL: {final_base_url}")
        print(f"   Max Workers: {max_workers}")

    def _generate_single(self, prompt):
        """Generate response for a single prompt"""
        try:
            # Build message format
            messages = [{"role": "user", "content": prompt}]
            
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # If logprobs is supported, add related parameters
            if self.logprobs_supported:
                params.update({
                    "logprobs": True,
                    "top_logprobs": self.top_logprobs
                })
            
            response = self.client.chat.completions.create(**params)
            
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                generated_text = choice.message.content
                
                # Build unified result object
                result = OpenAIResult(generated_text, choice, self.logprobs_supported)
                
                # Thread-safe log output
                with self.lock:
                    print(f"Tokens => Prompt: {response.usage.prompt_tokens}, "
                          f"Completion: {response.usage.completion_tokens}, "
                          f"Total: {response.usage.total_tokens}")
                
                return result
            else:
                with self.lock:
                    print(f"WARNING:  Empty response from OpenAI API")
                return OpenAIResult("", None, False)
                
        except Exception as e:
            with self.lock:
                print(f"WARNING:  Error calling OpenAI API: {e}")
            return OpenAIResult("", None, False)

    def generate(self, prompts):
        """Generate text using OpenAI API in parallel"""
        results = [None] * len(prompts)  # Pre-allocate result array to maintain order
        
        # Use ThreadPoolExecutor for parallelization
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks and remember their indices
            future_to_index = {
                executor.submit(self._generate_single, prompt): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            # Collect results with tqdm progress bar
            with tqdm(total=len(prompts), desc="OpenAI API parallel inference") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as exc:
                        with self.lock:
                            print(f"WARNING:  Prompt {idx} generated an exception: {exc}")
                        results[idx] = OpenAIResult("", None, False)
                    pbar.update(1)
        
        return results


def create_runner(model_name, temperature=0.0, api_key=None, base_url=None, max_workers=4, logprobs_supported=True):
    """Create corresponding inference engine based on model_name"""
    
    if '::' in model_name:
        # Parse format: backend::actual_model_name
        backend, actual_model = model_name.split('::', 1)
        backend = backend.lower()
        
        if backend == 'openai':
            runner = OpenAIRunner(actual_model, temperature, api_key, base_url, max_workers)
            runner.logprobs_supported = logprobs_supported
            return runner
        else:
            raise ValueError(f"Unsupported backend: {backend}. Supported: openai")
    else:
        # Default to vLLM (local model), no parallelization support (GPU already parallel)
        runner = VLLMRunner(model_name, temperature)
        runner.logprobs_supported = logprobs_supported
        return runner


def extract_model_prefix(model_name):
    """
    Extract model name prefix for file naming
    
    Rules:
    1. If in openai:: format, take the part after ::
    2. If contains /, only take the last part (e.g., minzl/toy_3550 -> toy_3550)
    3. Otherwise use directly
    
    Args:
        model_name: Full model name
        
    Returns:
        str: Model prefix for file naming
    """
    if '::' in model_name:
        # Handle backend::model format
        backend, actual_model = model_name.split('::', 1)
        if backend.lower() == 'openai':
            # For openai::model, use the model name after ::
            model_name = actual_model
        else:
            # For other backends, use the full format
            model_name = model_name.replace('::', '_')
    
    # Handle HuggingFace format paths (e.g., minzl/toy_3550)
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    
    # Replace other possible special characters with underscores
    model_name = model_name.replace('-', '_').replace('.', '_')
    
    return model_name


def construct_file_paths(input_dir, output_dir, dataset_name, split_seed, row_shuffle_seed, 
                        train_chunk_size, test_chunk_size, model_name):
    """Construct file paths based on parameters"""
    
    # Extract model prefix
    model_prefix = extract_model_prefix(model_name)
    
    # Unified subdirectory format: dataset_Sseed{split_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}
    subdir = f"{dataset_name}_Sseed{split_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}"
    
    # Input file path
    input_filename = f"{dataset_name}_Rseed{row_shuffle_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}.jsonl"
    input_jsonl = os.path.join(input_dir, dataset_name, subdir, input_filename)
    
    # Output file path - use model prefix at the beginning, remove _pred suffix
    output_filename = f"{model_prefix}@@{dataset_name}_Rseed{row_shuffle_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}.jsonl"
    output_jsonl = os.path.join(output_dir, dataset_name, subdir, output_filename)
    
    return input_jsonl, output_jsonl


def determine_input_output_paths(input_dir, output_dir, dataset_name=None, 
                                split_seed=42, row_shuffle_seed=123, train_chunk_size=600, 
                                test_chunk_size=7, model_name=None):
    """
    Intelligently determine input/output paths using new simplified logic
    
    Args:
        input_dir: Input path (could be .jsonl file or directory)
        output_dir: Output path (could be .jsonl file or directory)
        dataset_name: Dataset name (required when input/output is directory)
        model_name: Model name (required when output is directory)
        Other parameters: Parameters for building complex file paths
        
    Returns:
        tuple: (input_file_path, output_file_path, is_single_mode)
    """
    
    # 判断输入路径类型
    if input_dir.endswith('.jsonl'):
        # 输入是JSONL文件，直接使用
        input_file = input_dir
        is_single_mode = True
    else:
        # 输入是目录，需要构建复杂路径
        if not dataset_name:
            raise ValueError("When input_dir is a directory, dataset_name is required")
        
        # 使用复杂的路径构建逻辑
        input_file, _ = construct_file_paths(
            input_dir, "", dataset_name, split_seed, row_shuffle_seed,
            train_chunk_size, test_chunk_size, model_name or "dummy"
        )
        is_single_mode = False
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # 判断输出路径类型
    if output_dir.endswith('.jsonl'):
        # 输出是JSONL文件，直接使用并创建目录
        output_file = output_dir
        output_dirname = os.path.dirname(output_file)
        if output_dirname:  # 只有当目录名不为空时才创建
            os.makedirs(output_dirname, exist_ok=True)
    else:
        # 输出是目录，需要构建复杂路径
        if not dataset_name:
            raise ValueError("When output_dir is a directory, dataset_name is required")
        if not model_name:
            raise ValueError("When output_dir is a directory, model_name is required")
        
        # 使用复杂的路径构建逻辑
        _, output_file = construct_file_paths(
            "", output_dir, dataset_name, split_seed, row_shuffle_seed,
            train_chunk_size, test_chunk_size, model_name
        )
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    return input_file, output_file, is_single_mode


def main():
    parser = argparse.ArgumentParser(description='批量优化版 vLLM 推理脚本（支持API并行化）')
    
    # 统一的输入输出参数
    parser.add_argument("--input_dir", 
                        help="输入路径：可以是目录（需要配合dataset_names等参数自动构建文件路径）或直接的.jsonl文件路径")
    parser.add_argument("--output_dir", 
                        help="输出路径：可以是目录（自动构建文件路径）或直接的.jsonl文件路径（自动创建层级目录）")
    
    # 批量处理参数
    parser.add_argument("--dataset_names", 
                        help="数据集名称列表，用空格分隔，如：'bank heloc rl'")
    parser.add_argument("--row_shuffle_seeds", 
                        help="行打乱种子列表，用空格分隔，如：'40 41 42'")

    # 路径构建参数（当input_dir/output_dir是目录时用于构建复杂文件路径）
    parser.add_argument("--split_seed", type=int, default=42,
                       help="分割种子，用于数据集分割和确定性采样（默认：42）")
    parser.add_argument("--train_chunk_size", type=int, default=600,
                        help="训练块大小，每个prompt中包含的few-shot示例数量（默认：600）")
    parser.add_argument("--test_chunk_size", type=int, default=7,
                        help="测试块大小，聚合到单个prompt中进行LLM推理的项目数量（默认：7）")

    parser.add_argument('--labels', type=str, help='标签列表，用逗号分隔，如："0,1,2,3,4"')

   # 必需参数
    parser.add_argument('--model_name', type=str, required=True, 
                        help='模型名称或路径。支持格式：\n'
                             '  - 本地模型路径 (使用vLLM): /path/to/model\n'
                             '  - OpenAI API: openai::gpt-4o')
    
    # OpenAI API 参数（仅在使用OpenAI时需要）
    parser.add_argument('--api_key', type=str, help='OpenAI API密钥 (使用openai::模型时必需)')
    parser.add_argument('--base_url', type=str, help='OpenAI API基础URL (使用openai::模型时必需)')
    
    # 可选参数
    parser.add_argument('--temperature', type=float, default=0.0, help='采样温度')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数')
    parser.add_argument('--device_id', type=str, default="0", help='指定使用的GPU设备ID (默认: "0")')
    parser.add_argument('--force_overwrite', action='store_true', default=False,
                        help='如果设置，将直接删除已存在的输出文件而不询问')
    
    # 并行化参数
    parser.add_argument('--max_workers', type=int, default=4,
                        help='API并行推理的最大工作线程数（仅对openai::有效，默认：4）')
    parser.add_argument("--logprobs-supported", dest="logprobs_supported", 
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable logprobs support")
    
    args = parser.parse_args()
    
    # 设置 GPU 设备
    if args.device_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
        print(f"TARGET: 设置 GPU 设备: {args.device_id}")
    
    if not args.model_name:
        print("ERROR: Error: --model_name is required")
        sys.exit(1)
    
    # 验证必需参数
    if not args.input_dir or not args.output_dir:
        print("ERROR: Error: Both --input_dir and --output_dir are required")
        print("   --input_dir: 可以是目录或 .jsonl 文件路径")
        print("   --output_dir: 可以是目录或 .jsonl 文件路径")
        sys.exit(1)
    
    # 解析批量参数
    dataset_names = args.dataset_names.split() if args.dataset_names else []
    row_shuffle_seeds = [int(x) for x in args.row_shuffle_seeds.split()] if args.row_shuffle_seeds else []
    
    if not dataset_names or not row_shuffle_seeds:
        print("ERROR: Error: Both --dataset_names and --row_shuffle_seeds are required for batch processing")
        print("   --dataset_names: 'bank heloc rl'")
        print("   --row_shuffle_seeds: '40 41 42'")
        sys.exit(1)
    
    # 检查OpenAI相关参数
    if '::' in args.model_name:
        backend, _ = args.model_name.split('::', 1)
        backend = backend.lower()
        
        if backend == 'openai':
            # 检查API密钥和base_url
            api_key = args.api_key or os.getenv('OPENAI_API_KEY')
            base_url = args.base_url or os.getenv('OPENAI_BASE_URL')
            
            if not api_key:
                print("ERROR: Error: OpenAI API key is required when using openai:: backend")
                print("   Please provide --api_key or set OPENAI_API_KEY environment variable")
                sys.exit(1)
            
            if not base_url:
                print("ERROR: Error: OpenAI base URL is required when using openai:: backend")
                print("   Please provide --base_url or set OPENAI_BASE_URL environment variable")
                sys.exit(1)
    
    # 初始化推理器（只初始化一次！）
    print(f"STARTING: Initializing model once for all datasets and seeds...")
    
    # 检查并行化是否适用
    if '::' in args.model_name:
        backend, _ = args.model_name.split('::', 1)
        backend = backend.lower()
        if backend == 'openai':
            print(f"🔄 API并行化启用 - Max workers: {args.max_workers}")
    
    runner = create_runner(args.model_name, args.temperature, args.api_key, args.base_url, args.max_workers, args.logprobs_supported)
    
    # 解析用户提供的标签
    user_labels = None
    if args.labels:
        # 解析逗号分隔的标签字符串
        label_list = [label.strip() for label in args.labels.split(',')]
        # 排序标签：数字优先，然后字母
        user_labels = sorted(label_list, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else float('inf'), x))
        print(f"INFO: Using user-provided labels: {user_labels}")
    else:
        print(f"INFO: No labels provided, will auto-infer from file or label_transform_info.json")
    
    # 计算总任务数
    total_tasks = len(dataset_names) * len(row_shuffle_seeds)
    current_task = 0
    successful_tasks = 0
    failed_tasks = 0
    
    print(f"INFO: Batch processing summary:")
    print(f"   Datasets: {dataset_names}")
    print(f"   Row shuffle seeds: {row_shuffle_seeds}")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Model will be loaded only once! TARGET:")
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
                    input_file, output_file, is_single_mode = determine_input_output_paths(
                        args.input_dir, args.output_dir, dataset_name,
                        args.split_seed, row_shuffle_seed, args.train_chunk_size,
                        args.test_chunk_size, args.model_name
                    )
                    
                    print(f"INPUT: File paths:")
                    print(f"   Input:  {input_file}")
                    print(f"   Output: {output_file}")
                    
                    # 处理文件
                    success = runner.process_file(
                        input_file, 
                        output_file, 
                        max_samples=args.max_samples,
                        user_labels=user_labels,
                        force_overwrite=args.force_overwrite
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
