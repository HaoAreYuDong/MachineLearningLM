import os
import json
import math
import argparse
from pathlib import Path
import sys
import re
import concurrent.futures
from threading import Lock

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

from tqdm import tqdm

from dl_utils import BaseRunner, exp_safe


class OpenAIResult:
    
    def __init__(self, text, choice, logprobs_supported=True):
        self.text = text
        self.choice = choice
        self.logprobs_supported = logprobs_supported
        
        self.outputs = [self]
        
        self.logprobs_data = None
        if logprobs_supported and choice and hasattr(choice, 'logprobs') and choice.logprobs:
            self.logprobs_data = self._process_logprobs(choice.logprobs)
    
    def _process_logprobs(self, logprobs):
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
    
    def __init__(self, model_name, temperature=0.0):
        super().__init__(model_name, temperature)
        
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

        print(f"🌟 Initializing vLLM model: {model_name}")

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
        
        print(f"✅ vLLM Model loaded successfully!")

    def generate(self, prompts):
        return self.client.generate(prompts, self.sampling_params)


class OpenAIRunner(BaseRunner):
    
    def __init__(self, model_name, temperature=0.0, api_key=None, base_url=None, max_workers=4):
        super().__init__(model_name, temperature)

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

        final_api_key = api_key or os.getenv('OPENAI_API_KEY')
        final_base_url = base_url or os.getenv('OPENAI_BASE_URL')
        
        if not final_api_key:
            raise ValueError("API key is required for OpenAI. Please provide --api_key or set OPENAI_API_KEY environment variable")
        if not final_base_url:
            raise ValueError("Base URL is required for OpenAI. Please provide --base_url or set OPENAI_BASE_URL environment variable")
        
        self.client = openai.OpenAI(api_key=final_api_key, base_url=final_base_url)
        self.logprobs_supported = True
        
        self.lock = Lock()
        
        print(f"✅ OpenAI API initialized successfully!")
        print(f"   API Key: {final_api_key[:8]}...")
        print(f"   Base URL: {final_base_url}")
        print(f"   Max Workers: {max_workers}")

    def _generate_single(self, prompt):
        try:
            messages = [{"role": "user", "content": prompt}]
            
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            if self.logprobs_supported:
                params.update({
                    "logprobs": True,
                    "top_logprobs": self.top_logprobs
                })
            
            response = self.client.chat.completions.create(**params)
            
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                generated_text = choice.message.content
                
                result = OpenAIResult(generated_text, choice, self.logprobs_supported)
                
                with self.lock:
                    print(f"Tokens => Prompt: {response.usage.prompt_tokens}, "
                          f"Completion: {response.usage.completion_tokens}, "
                          f"Total: {response.usage.total_tokens}")
                
                return result
            else:
                with self.lock:
                    print(f"⚠️  Empty response from OpenAI API")
                return OpenAIResult("", None, False)
                
        except Exception as e:
            with self.lock:
                print(f"⚠️  Error calling OpenAI API: {e}")
            return OpenAIResult("", None, False)

    def generate(self, prompts):
        results = [None] * len(prompts)  
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._generate_single, prompt): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            with tqdm(total=len(prompts), desc="OpenAI API parallel inference") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as exc:
                        with self.lock:
                            print(f"⚠️  Prompt {idx} generated an exception: {exc}")
                        results[idx] = OpenAIResult("", None, False)
                    pbar.update(1)
        
        return results




def create_runner(model_name, temperature=0.0, api_key=None, base_url=None, max_workers=4, logprobs_supported=True):
    
    if '::' in model_name:
        backend, actual_model = model_name.split('::', 1)
        backend = backend.lower()
        
        if backend == 'openai':
            runner = OpenAIRunner(actual_model, temperature, api_key, base_url, max_workers)
            runner.logprobs_supported = logprobs_supported
            return runner
        else:
            raise ValueError(f"Unsupported backend: {backend}. Supported: openai")
    else:
        runner = VLLMRunner(model_name, temperature)
        runner.logprobs_supported = logprobs_supported
        return runner


def extract_model_prefix(model_name):
    """
    Extract model name prefix for file naming

    Rules:
    1. For `openai::` formats, use the part after the prefix
    2. If `/` is present, use only the last segment (e.g. `minzl/toy_3550` → `toy_3550`)
    3. Use original name in all other cases

    Args:
        model_name: Full model name string

    Returns:
        str: Model prefix for file naming
    """
    if '::' in model_name:
        backend, actual_model = model_name.split('::', 1)
        if backend.lower() == 'openai':
            model_name = actual_model
        else:
            model_name = model_name.replace('::', '_')
    
    # Handling HuggingFace format paths
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    model_name = model_name.replace('-', '_').replace('.', '_')
    
    return model_name


def construct_file_paths(input_dir, output_dir, dataset_name, split_seed, row_shuffle_seed, 
                        train_chunk_size, test_chunk_size, model_name):
    
    model_prefix = extract_model_prefix(model_name)
    subdir = f"{dataset_name}_Sseed{split_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}"
    
    input_filename = f"{dataset_name}_Rseed{row_shuffle_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}.jsonl"
    input_jsonl = os.path.join(input_dir, dataset_name, subdir, input_filename)
    
    output_filename = f"{model_prefix}@@{dataset_name}_Rseed{row_shuffle_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}.jsonl"
    output_jsonl = os.path.join(output_dir, dataset_name, subdir, output_filename)
    
    return input_jsonl, output_jsonl


def determine_input_output_paths(input_dir, output_dir, dataset_name=None, 
                                split_seed=42, row_shuffle_seed=123, train_chunk_size=600, 
                                test_chunk_size=7, model_name=None):
    """
    Intelligently determines input and output paths using a new simplified logic.

    Args:
        input_dir: Input path (could be a .jsonl file or a directory)
        output_dir: Output path (could be a .jsonl file or a directory)
        dataset_name: Dataset name (required when input/output is a directory)
        model_name: Model name (required when output is a directory)
        Other parameters: Used for constructing complex file paths

    Returns:
        tuple: (input_file_path, output_file_path, is_single_mode)
    """
    
    if input_dir.endswith('.jsonl'):
        input_file = input_dir
        is_single_mode = True
    else:
        if not dataset_name:
            raise ValueError("When input_dir is a directory, dataset_name is required")
        
        input_file, _ = construct_file_paths(
            input_dir, "", dataset_name, split_seed, row_shuffle_seed,
            train_chunk_size, test_chunk_size, model_name or "dummy"
        )
        is_single_mode = False
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if output_dir.endswith('.jsonl'):
        output_file = output_dir
        output_dirname = os.path.dirname(output_file)
        if output_dirname:
            os.makedirs(output_dirname, exist_ok=True)
    else:
        if not dataset_name:
            raise ValueError("When output_dir is a directory, dataset_name is required")
        if not model_name:
            raise ValueError("When output_dir is a directory, model_name is required")
        
        _, output_file = construct_file_paths(
            "", output_dir, dataset_name, split_seed, row_shuffle_seed,
            train_chunk_size, test_chunk_size, model_name
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    return input_file, output_file, is_single_mode


def main():
    parser = argparse.ArgumentParser(description='Batch Optimized vLLM Inference Script')
    
    # Unified input/output parameters
    parser.add_argument("--input_dir", 
                        help="Input path: can be a directory (requires automatic file path construction with parameters like dataset_names) or a direct .jsonl file path")
    parser.add_argument("--output_dir", 
                        help="Output path: can be a directory (automatic file path construction) or a direct .jsonl file path (automatically creates hierarchical directories)")
    
    # Batch processing parameters
    parser.add_argument("--dataset_names", 
                        help="Dataset name list, separated by spaces, e.g.: 'bank heloc rl'")
    parser.add_argument("--row_shuffle_seeds", 
                        help="Row shuffle seed list, separated by spaces, e.g.: '40 41 42'")

    # Path construction parameters (used when input_dir/output_dir are directories for complex file path building)
    parser.add_argument("--split_seed", type=int, default=42,
                       help="Split seed for dataset splitting and deterministic sampling (default: 42)")
    parser.add_argument("--train_chunk_size", type=int, default=600,
                        help="Training chunk size, number of few-shot examples included in each prompt (default: 600)")
    parser.add_argument("--test_chunk_size", type=int, default=7,
                        help="Test chunk size, number of items aggregated into a single prompt for LLM inference (default: 7)")

    parser.add_argument('--labels', type=str, help='Label list, comma-separated, e.g.: "0,1,2,3,4"')

   # Required parameters
    parser.add_argument('--model_name', type=str, required=True, 
                        help='Model name or path. Supported formats:\n'
                             '  - Local model path (using vLLM): /path/to/model\n'
                             '  - OpenAI API: openai::gpt-4o\n')
    
    # OpenAI API parameters (required only when using OpenAI)
    parser.add_argument('--api_key', type=str, help='OpenAI API key (required when using openai:: models)')
    parser.add_argument('--base_url', type=str, help='OpenAI API base URL (required when using openai:: models)')
    
    # Optional parameters
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples')
    parser.add_argument('--device_id', type=str, default="0", help='Specify GPU device ID to use (default: "0")')
    parser.add_argument('--force_overwrite', action='store_true', default=False,
                        help='If set, will directly delete existing output files without prompting')
    
    # Parallelization parameters
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads for API parallel inference (only effective for openai::models, default: 4)')
    parser.add_argument("--logprobs-supported", dest="logprobs_supported", 
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable logprobs support")
    
    args = parser.parse_args()
    
    if args.device_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
        print(f"🎯 Set GPU device: {args.device_id}")

    if not args.model_name:
        print("❌ Error: --model_name is required")
        sys.exit(1)

    if not args.input_dir or not args.output_dir:
        print("❌ Error: Both --input_dir and --output_dir are required")
        print("   --input_dir: Can be a directory or .jsonl file path")
        print("   --output_dir: Can be a directory or .jsonl file path")
        sys.exit(1)
    
    dataset_names = args.dataset_names.split() if args.dataset_names else []
    row_shuffle_seeds = [int(x) for x in args.row_shuffle_seeds.split()] if args.row_shuffle_seeds else []
    
    if not dataset_names or not row_shuffle_seeds:
        print("❌ Error: Both --dataset_names and --row_shuffle_seeds are required for batch processing")
        print("   --dataset_names: 'bank heloc rl'")
        print("   --row_shuffle_seeds: '40 41 42'")
        sys.exit(1)
    
    if '::' in args.model_name:
        backend, _ = args.model_name.split('::', 1)
        backend = backend.lower()
        
        if backend == 'openai':
            api_key = args.api_key or os.getenv('OPENAI_API_KEY')
            base_url = args.base_url or os.getenv('OPENAI_BASE_URL')
            
            if not api_key:
                print("❌ Error: OpenAI API key is required when using openai:: backend")
                print("   Please provide --api_key or set OPENAI_API_KEY environment variable")
                sys.exit(1)
            
            if not base_url:
                print("❌ Error: OpenAI base URL is required when using openai:: backend")
                print("   Please provide --base_url or set OPENAI_BASE_URL environment variable")
                sys.exit(1)
    
    print(f"🚀 Initializing model once for all datasets and seeds...")
    
    # Check if parallelization is applicable
    if '::' in args.model_name:
        backend, _ = args.model_name.split('::', 1)
        backend = backend.lower()
        if backend == 'openai':
            print(f"🔄 API parallelization enabled - Max workers: {args.max_workers}")
    
    runner = create_runner(args.model_name, args.temperature, args.api_key, args.base_url, args.max_workers, args.logprobs_supported)
    
    user_labels = None
    if args.labels:
        label_list = [label.strip() for label in args.labels.split(',')]
        user_labels = sorted(label_list, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else float('inf'), x))
        print(f"📊 Using user-provided labels: {user_labels}")
    else:
        print(f"📊 No labels provided, will auto-infer from file or label_transform_info.json")
    
    total_tasks = len(dataset_names) * len(row_shuffle_seeds)
    current_task = 0
    successful_tasks = 0
    failed_tasks = 0
    
    print(f"📊 Batch processing summary:")
    print(f"   Datasets: {dataset_names}")
    print(f"   Row shuffle seeds: {row_shuffle_seeds}")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Model will be loaded only once! 🎯")
    print("")
    
    try:
        for dataset_name in dataset_names:
            for row_shuffle_seed in row_shuffle_seeds:
                current_task += 1
                
                print(f"┌─────────────────────────────────────────────────────────────────────────────┐")
                print(f"│ Task {current_task:2d}/{total_tasks:2d} │ Dataset: {dataset_name:10s} │ Seed: {row_shuffle_seed:4d} │")
                print(f"└─────────────────────────────────────────────────────────────────────────────┘")
                
                try:
                    input_file, output_file, is_single_mode = determine_input_output_paths(
                        args.input_dir, args.output_dir, dataset_name,
                        args.split_seed, row_shuffle_seed, args.train_chunk_size,
                        args.test_chunk_size, args.model_name
                    )
                    
                    print(f"📂 File paths:")
                    print(f"   Input:  {input_file}")
                    print(f"   Output: {output_file}")
                    
                    success = runner.process_file(
                        input_file, 
                        output_file, 
                        max_samples=args.max_samples,
                        user_labels=user_labels,
                        force_overwrite=args.force_overwrite
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
