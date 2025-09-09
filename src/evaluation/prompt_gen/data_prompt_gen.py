"""
Refactored prompt generation main script.
This file contains the orchestration (I/O, parallel processing) and uses
extension hooks in `prompt_utils.py` for format/normalization/token counting.
"""

import os
import json
import argparse
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import shutil

import pandas as pd
import tiktoken

# Local extensions (same-folder)
# Make imports robust so the file can be executed either as a module or as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from . import prompt_utils as ext
except Exception:
    import prompt_utils as ext


# --- EXTERNAL HOOK: place at top to make it obvious this is user-replaceable ---
def external_prompt_builder(X_train_df, X_test_df, y_train, y_test, args):
    """
    External customizable prompt builder (hook interface).

    Signature: (X_train_df, X_test_df, y_train, y_test, args)
    - `args` may be a dict or an argparse Namespace; expected keys/options include
      'normalization', 'include_feature_descriptions', 'prompt_format_style'.

    This function is intended to be a user-replaceable external interface (hook)
    for customizing prompt creation. Modify this function to add/remove features
    or change prompt formatting.
    """
    def _get_opt(obj, key, default=None):
        try:
            # dict-like
            return obj.get(key, default)
        except Exception:
            # namespace-like
            return getattr(obj, key, default)

    normalization = _get_opt(args, 'normalization', False)
    include_feature_descriptions = _get_opt(args, 'include_feature_descriptions', False)
    prompt_format_style = _get_opt(args, 'prompt_format_style', 'concat')

    feature_names = list(X_train_df.columns)

    # Normalize / cast features
    if normalization:
        X_train_arr, X_test_arr = ext.normalize_feature_arrays(X_train_df, X_test_df)
    else:
        X_train_arr, X_test_arr = ext.cast_int_non_normalized(X_train_df, X_test_df)

    # Ensure y arrays are 1-d
    y_test = ext.ensure_1d(y_test)
    y_train = ext.ensure_1d(y_train)

    # Prepare train/test lines according to format
    train_lines, test_lines = ext.format_lines(X_train_arr, y_train, X_test_arr, feature_names, prompt_format_style)

    labels_json = ext.create_labels_json(y_test)

    system_prompt = "You are an AI assistant. Your task is supervised classification."
    user_prompt = ext.build_user_prompt(X_test_arr.shape[1], sorted([int(x) for x in set(y_train)]), 
                                        feature_names,
                                        train_lines, 
                                        test_lines, 
                                        prompt_format_style,
                                        include_feature_descriptions)

    prompt_obj = ext.build_full_prompt(system_prompt, user_prompt, labels_json)

    return prompt_obj, len(train_lines), len(test_lines)


# Keep process_single_chunk as a top-level function so it remains easily picklable
# for ProcessPoolExecutor workers. This function calls the external hook above.
def process_single_chunk(args):
    """Process a single chunk file and return statistics/prompt string."""
    (split_data_dir, fname, normalization,
     include_feature_descriptions, prompt_format_style, row_shuffle_seed) = args

    # Add Rseed layer to the split_data_dir path
    rseed_data_dir = os.path.join(split_data_dir, f"Rseed{row_shuffle_seed}")

    # Build file paths from rseed_data_dir + standard subfolders
    x_test_file = os.path.join(rseed_data_dir, 'X_test', fname)
    X_train_file = os.path.join(rseed_data_dir, 'X_train', fname)
    y_test_file = os.path.join(rseed_data_dir, 'y_test', fname)
    y_train_file = os.path.join(rseed_data_dir, 'y_train', fname)

    if not all(os.path.exists(p) for p in [x_test_file, X_train_file, y_train_file, y_test_file]):
        return fname, 0, 0, 0, 0, 0, None

    X_test_df = pd.read_csv(x_test_file)
    X_train_df = pd.read_csv(X_train_file)
    y_test = pd.read_csv(y_test_file).squeeze()
    y_train = pd.read_csv(y_train_file).squeeze()


    # Build prompt via user-customizable helper
    user_args = {
        'normalization': normalization,
        'include_feature_descriptions': include_feature_descriptions,
        'prompt_format_style': prompt_format_style
    }
    prompt_obj, few_shot, test_examples = external_prompt_builder(
        X_train_df, X_test_df, y_train, y_test, user_args
    )

    # Token counting and size check
    num_tokens = ext.get_encoder(prompt_obj)
    if num_tokens > 110000:
        return fname, 1, 0, 0, 0, num_tokens, None

    prompt_data = json.dumps(prompt_obj, ensure_ascii=False) + "\n"
    return fname, 0, 1, few_shot, test_examples, num_tokens, prompt_data


# Encapsulate orchestration into a class so main simply calls PromptGenerator.run()
class PromptGenerator:
    """Encapsulates dataset prompt generation orchestration."""

    def __init__(
        self,
        input_base_dir,
        output_dir,
        dataset_name,
        split_seed,
        row_shuffle_seed,
        chunk_size_train,
        chunk_size_test,
        normalization=True,
        include_feature_descriptions=False,
        prompt_format_style="concat",
        max_workers=8,
        force_overwrite=False,
    ):
        self.input_base_dir = input_base_dir
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        # CLI-aligned seeds: split_seed and row_shuffle_seed
        self.split_seed = split_seed
        # Prefer explicit row_shuffle_seed for folder naming; fallback to split_seed
        self.row_shuffle_seed = row_shuffle_seed if row_shuffle_seed is not None else split_seed
        self.chunk_size_train = chunk_size_train
        self.chunk_size_test = chunk_size_test
        self.normalization = normalization
        self.include_feature_descriptions = include_feature_descriptions
        self.prompt_format_style = prompt_format_style
        self.max_workers = max_workers
        self.force_overwrite = force_overwrite

    def prepare_dataset_output(self, split_data_dir):
        """
        Prepare and validate the output directory for a dataset split.

        Returns (dataset_output_dir, list_of_csv_paths)
        """
        rseed_data_dir = os.path.join(split_data_dir, f"Rseed{self.row_shuffle_seed}")
        X_test_dir = os.path.join(rseed_data_dir, 'X_test')

        # Validate test CSVs
        if not os.path.isdir(X_test_dir) or not any(fname.lower().endswith('.csv') for fname in os.listdir(X_test_dir)):
            msg_lines = [
                f"ERROR: No test CSVs found in: {X_test_dir}",
                "Aborting serialization and exiting."
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

            sys.exit(1)

        # Compute dataset suffix and output folder to mirror data_chunk_prep layout
        output_folder_suffix = f"{self.dataset_name}_Sseed{self.split_seed}_trainSize{self.chunk_size_train}_testSize{self.chunk_size_test}"

        dataset_output_dir = os.path.join(self.output_dir, self.dataset_name, output_folder_suffix)

        # If target dataset dir exists and contains files, ask or use force flag
        # Ensure base output directory exists (no per-subdir creation needed for prompt generation)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Collect CSV file paths from the rseed_data_dir's X_test folder
        csv_files = [os.path.join(rseed_data_dir, 'X_test', fname)
                     for fname in os.listdir(os.path.join(rseed_data_dir, 'X_test'))
                     if fname.lower().endswith('.csv')]

        return dataset_output_dir, csv_files

    def process_and_serialize(self):
        # Normalize input/output paths so code behaves consistently regardless of CWD
        self.input_base_dir = os.path.abspath(self.input_base_dir)
        self.output_dir = os.path.abspath(self.output_dir)

        # Use row_shuffle_seed for dataset folder naming to match data_chunk_prep behavior
        input_folder_suffix = f"{self.dataset_name}_Sseed{self.split_seed}_trainSize{self.chunk_size_train}_testSize{self.chunk_size_test}"

        # Compute split_data_dir: location where data_chunk_prep wrote X_train/X_test/y_train/y_test
        split_data_dir = os.path.join(self.input_base_dir, self.dataset_name, input_folder_suffix)

        # Delegate output directory preparation and CSV collection to prepare_dataset_output
        dataset_output_dir, csv_files = self.prepare_dataset_output(split_data_dir)

        output_file = os.path.join(
            dataset_output_dir,
            f"{self.dataset_name}_Rseed{self.row_shuffle_seed}_trainSize{self.chunk_size_train}_testSize{self.chunk_size_test}.jsonl"
         )
        summary_file = os.path.join(dataset_output_dir, f'usable_counts_{self.dataset_name}_Rseed{self.row_shuffle_seed}_trainSize{self.chunk_size_train}_testSize{self.chunk_size_test}.txt')

        # Check if the specific output .jsonl file already exists
        if os.path.exists(output_file):
            msg_lines = [
                "ERROR: Target output file already exists.",
                "To avoid accidental overwrite, you can choose to delete and recreate it.",
                f"Target: {output_file}",
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
            if self.force_overwrite or (not interactive_mode):
                choice = 'y'
                if self.force_overwrite:
                    print("--force_overwrite flag detected: proceeding to delete and recreate the file.")
                else:
                    print("Non-interactive environment detected: defaulting to delete and recreate (no prompt).")
            else:
                try:
                    choice = input("Delete and recreate the existing file? [y/N]: ").strip().lower()
                except Exception:
                    print("No input available or input interrupted. Aborting.")
                    sys.exit(1)

            if choice in ("y", "yes"):
                try:
                    os.remove(output_file)
                    print(f"Removed existing file: {output_file}")
                except Exception as exc:
                    print(f"Failed to remove file {output_file}: {exc}")
                    sys.exit(1)
            else:
                print("Operation aborted by user. Exiting without changes.")
                sys.exit(0)

        print(f"[PromptGen] Processing dataset: {self.dataset_name}")
        print(f"[PromptGen] Normalization: {'Enabled' if self.normalization else 'Disabled'}")

        with open(output_file, 'w', encoding='utf-8') as f:
            print(f"[Info] Started generating prompts for {self.dataset_name}...")

            usable_count = 0
            discarded_count = 0
            total_few_shot_examples = 0
            total_test_examples = 0
            total_tokens = 0
            total_samples = 0

            # Prepare arguments for parallel processing
            chunk_args = []
            for X_test_file in csv_files:
                fname = os.path.basename(X_test_file)
                args = (split_data_dir, fname, self.normalization, self.include_feature_descriptions, self.prompt_format_style, self.row_shuffle_seed)
                chunk_args.append(args)

            # Determine number of workers according to CLI policy
            cpu_count = multiprocessing.cpu_count()
            if self.max_workers == 1:
                # Serial processing
                print(f"[Optimization] Running serial processing over {len(chunk_args)} chunks...")

                prompt_data_list = []
                completed = 0
                with tqdm(total=len(chunk_args),
                          desc=f"Generating prompts for {self.dataset_name} (seed={self.row_shuffle_seed}, train_size={self.chunk_size_train})") as pbar:
                    for args in chunk_args:
                        fname, discarded, usable, few_shot, test_examples, tokens, prompt_data = process_single_chunk(args)
                        completed += 1

                        discarded_count += discarded
                        usable_count += usable
                        total_few_shot_examples += few_shot
                        total_test_examples += test_examples
                        total_tokens += tokens
                        total_samples += 1 if (discarded + usable) > 0 else 0

                        if usable > 0 and prompt_data:
                            prompt_data_list.append((fname, prompt_data))

                        pbar.update(1)
                        pbar.set_postfix({
                            'usable': usable_count,
                            'discarded': discarded_count,
                            'rate': f'{completed/len(chunk_args)*100:.1f}%'
                        })
            else:
                if self.max_workers == -1:
                    actual_workers = min(cpu_count, len(chunk_args))
                else:
                    actual_workers = min(cpu_count, len(chunk_args), self.max_workers)

                actual_workers = max(1, actual_workers)

                print(f"[Optimization] Using {actual_workers} parallel workers for {len(chunk_args)} chunks...")

                # Parallel processing
                prompt_data_list = []
                with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                    future_to_chunk = {executor.submit(process_single_chunk, args): args[1] for args in chunk_args}

                    completed = 0
                    batch_size = max(1, len(chunk_args) // 100)

                    with tqdm(total=len(chunk_args),
                             desc=f"Generating prompts for {self.dataset_name} (seed={self.row_shuffle_seed}, train_size={self.chunk_size_train})") as pbar:
                        batch_completed = 0
                        for future in as_completed(future_to_chunk):
                            fname, discarded, usable, few_shot, test_examples, tokens, prompt_data = future.result()
                            completed += 1
                            batch_completed += 1

                            discarded_count += discarded
                            usable_count += usable
                            total_few_shot_examples += few_shot
                            total_test_examples += test_examples
                            total_tokens += tokens
                            total_samples += 1 if (discarded + usable) > 0 else 0

                            if usable > 0 and prompt_data:
                                prompt_data_list.append((fname, prompt_data))

                            if batch_completed >= batch_size or completed == len(chunk_args):
                                pbar.update(batch_completed)
                                pbar.set_postfix({
                                    'usable': usable_count,
                                    'discarded': discarded_count,
                                    'rate': f'{completed/len(chunk_args)*100:.1f}%'
                                })
                                batch_completed = 0

            # Batch write prompt data
            print(f"[Info] Writing {len(prompt_data_list)} usable prompts to output file...")
            for fname, prompt_data in sorted(prompt_data_list, key=lambda x: x[0]):
                f.write(prompt_data)

            avg_few_shot = total_few_shot_examples / usable_count if usable_count else 0
            avg_test = total_test_examples / usable_count if usable_count else 0
            avg_tokens = total_tokens / total_samples if total_samples else 0

            print(f"Dataset '{self.dataset_name}': {usable_count} usable, {discarded_count} discarded")
            with open(summary_file, 'w', encoding='utf-8') as sf:
                sf.write(f"{self.dataset_name}: {usable_count} usable, {discarded_count} discarded\n")
                sf.write(f"  Avg few-shot: {avg_few_shot:.1f}, Avg test: {avg_test:.1f}, Avg tokens/sample: {avg_tokens:.1f}\n")
                sf.write(f"  Normalization: {'Enabled' if self.normalization else 'Disabled'}\n")
                sf.write(f"  Feature descriptions: {'Enabled' if self.include_feature_descriptions else 'Disabled'}\n")
                sf.write(f"  Format style: {self.prompt_format_style}\n")

    def run(self):
        """Convenience entrypoint: run the prompt generation flow."""
        self.process_and_serialize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Align CLI with data_chunk_prep parameter names and order
    parser.add_argument("--input_dir", required=True,
                        help="Base folder containing datasets (expects {input_dir}/{dataset_name}/{dataset_suffix})")
    parser.add_argument("--output_dir", required=True,
                        help="Base output directory")
    parser.add_argument("--dataset_name", required=True,
                        help="Name of the dataset (e.g., bank)")

    parser.add_argument("--split_seed", type=int, default=42,
                       help="Seed used for dataset suffix selection and deterministic sampling (default: 42)")
    # UNUSED: kept for CLI compatibility with other scripts
    parser.add_argument("--row_shuffle_seed", type=int, default=123, required=False,
                        help="(Unused here) row shuffle seed for internal sampling")  # UNUSED

    parser.add_argument("--train_chunk_size", type=int, default=600,
                        help="Number of few-shot examples (shots) included in each prompt for training chunks")
    parser.add_argument("--test_chunk_size", type=int, default=7,
                        help="Number of items aggregated into a single prompt for LLM inference (per-prompt inference batch size)")


    parser.add_argument("--max_workers", type=int, default=64,
                        help="Max workers parameter. If set to 1, processing will run serially. If -1, use min(cpu_count, n_chunks). Otherwise use min(cpu_count, n_chunks, value). Default=64")

    parser.add_argument("--force_overwrite", action="store_true", default=False,
                        help="If the output directory exists and is not empty, this flag allows overwriting it without prompt. Use with caution!")

    # Prompt generation specific options
    parser.add_argument("--normalization", type=lambda v: v.lower() in ("true", "1", "yes", "y"), default=True, required=False,
                        help="Apply normalization to the data (default: True)")
    parser.add_argument("--include_feature_descriptions", type=lambda v: v.lower() in ("true", "1", "yes", "y"), default=False, required=False,
                        help="Include feature descriptions in the prompt (default: False)")
    parser.add_argument("--prompt_format_style", type=str, default="concat", choices=["concat", "tabllm"],
                        help="Data formatting style: 'concat' or 'tabllm' (default: concat)")

    args = parser.parse_args()

    print(f"[PromptGen] Starting prompt generation (aligned CLI)...")
    generator = PromptGenerator(
        args.input_dir,
        args.output_dir,
        args.dataset_name,
        args.split_seed,
        args.row_shuffle_seed,
        args.train_chunk_size,
        args.test_chunk_size,
        args.normalization,
        args.include_feature_descriptions,
        args.prompt_format_style,
        args.max_workers,
        args.force_overwrite
     )

    generator.run()

    try:
        split_dir = args.input_dir
        dataset_suffix = f"{args.dataset_name}_Sseed{args.split_seed}_trainSize{args.train_chunk_size}_testSize{args.test_chunk_size}"
        source_label_file = os.path.join(split_dir, args.dataset_name, dataset_suffix, "label_transform_info.json")
        
        output_subdir = f"{args.dataset_name}_Sseed{args.split_seed}_trainSize{args.train_chunk_size}_testSize{args.test_chunk_size}"
        target_dir = os.path.join(args.output_dir, args.dataset_name, output_subdir)
        target_label_file = os.path.join(target_dir, "label_transform_info.json")
        
        if os.path.exists(source_label_file):
            os.makedirs(target_dir, exist_ok=True)
            
            if os.path.exists(target_label_file):
                pass
            else:
                # 复制文件
                shutil.copy2(source_label_file, target_label_file)
                print(f"[PromptGen] ✅ Copied label_transform_info.json to: {target_label_file}")
        else:
            print(f"[PromptGen] ⚠️  Warning: Source label_transform_info.json not found: {source_label_file}")
            
    except Exception as e:
        print(f"[PromptGen] ⚠️  Warning: Failed to copy label_transform_info.json: {e}")

    print(f"[PromptGen] Prompt generation completed successfully!")
