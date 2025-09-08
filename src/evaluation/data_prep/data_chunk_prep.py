"""
Refactored main data preparation script that relies on a small set of
extension functions (see `data_utils.py` in the same folder).

This module preserves the original behaviour of the script while keeping
extension points (column shuffling, mapping, pseudo class addition,
label discretization) encapsulated in `data_utils.py` to make it easy to add
more extensions later.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import shutil

# Ensure local package import works even when executed from another CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import data_utils as ext  # local extension module


def build_dataset_context(all_original_labels: pd.Series, output_dir: str):
    """
    Prepare dataset-level label mapping (use 'from_zero' by default) and
    optionally save minimal mapping info to disk. Pseudo-class support has
    been removed; this function now returns only the label_mapping and the
    mapping_path for backward compatibility where mapping_path is used.

    Returns: (label_mapping, mapping_path)
    """
    # Build label mapping based on requested transform mode; ignore unique_labels
    label_mapping, _ = ext.build_label_mapping(all_original_labels, "from_zero")

    # Persist minimal mapping info (data_utils.save_label_transform_info has been
    # updated to only persist essential fields). Pass an empty list for
    # unique_labels to avoid computing/storing unnecessary data.
    mapping_path = ext.save_label_transform_info(output_dir=output_dir,
                                                 label_transform_mode="from_zero",
                                                 add_pseudo_class_flag=False,
                                                 unique_labels=[],
                                                 label_mapping=label_mapping,
                                                 pseudo_label=None)

    # Return only the essential results
    return label_mapping, mapping_path


# Chunk-level transformations: apply per-chunk (row/col) transforms.
def apply_chunk_transforms(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_raw: pd.Series,
    y_test_raw: pd.Series,
    label_mapping,
    row_shuffle_seed: int,
    shuffle_columns: bool
):
    """
    Centralize per-chunk extension hooks so all optional, dataset-specific
    transformations are applied in one place. This makes it easy to add
    further extension checks later.

    Returns: X_train, X_test, y_train, y_test
    """
    # Apply label mapping (delegated to extensions)
    y_train, y_test = ext.apply_label_mapping(y_train_raw, y_test_raw, label_mapping)

    # Shuffle column order based on chunk-level shuffle seed for consistency (if enabled)
    if shuffle_columns:
        X_train, X_test = ext.col_shuffle(X_train, X_test, row_shuffle_seed)

    return X_train, X_test, y_train, y_test


def process_single_chunk(args):
    """
    Process a single chunk - designed for parallel execution. The arguments
    tuple mirrors the previous implementation to keep compatibility with the
    existing invocation logic.
    """
    (i, dataset_output_dir, label_mapping, row_shuffle_seed, shuffle_columns, test_chunk, train_chunk) = args

    # Extract features and labels with minimal overhead
    X_train = train_chunk.iloc[:, :-1]
    y_train_raw = train_chunk.iloc[:, -1]
    X_test = test_chunk.iloc[:, :-1]
    y_test_raw = test_chunk.iloc[:, -1]

    # Apply all configured per-chunk extension hooks in one place
    X_train, X_test, y_train, y_test = apply_chunk_transforms(
        X_train,
        X_test,
        y_train_raw,
        y_test_raw,
        label_mapping,
        row_shuffle_seed,
        shuffle_columns
    )

    # Create row shuffle seed specific directory
    rseed_dir = os.path.join(dataset_output_dir, f"Rseed{row_shuffle_seed}")

    # Save files with optimized IO
    X_train.to_csv(os.path.join(rseed_dir, "X_train", f"{i}.csv"), index=False)
    X_test.to_csv(os.path.join(rseed_dir, "X_test", f"{i}.csv"), index=False)

    # Convert to DataFrame for consistency with existing pipeline
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)

    y_train_df.to_csv(os.path.join(rseed_dir, "y_train", f"{i}.csv"), index=False)
    y_test_df.to_csv(os.path.join(rseed_dir, "y_test", f"{i}.csv"), index=False)

    return i, "success"


class DataChunkPrepRunner:
    """Class-based wrapper for the original data chunk preparation script.

    The runner preserves the original logic but provides a run() API and a
    configurable max_workers parameter exposed through argparse.

    max_workers_param semantics:
    - 1: process chunks serially in a simple for-loop.
    - -1: use min(multiprocessing.cpu_count(), len(chunk_args)).
    - otherwise: use min(multiprocessing.cpu_count(), len(chunk_args), max_workers_param).
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        dataset_name: str,
        split_seed: int = 42,
        row_shuffle_seed: int = 123,
        train_chunk_size: int = 32,
        test_chunk_size: int = 7,
        test_size: float = 0.2,
        shuffle_columns: bool = False,
        max_workers_param: int = 64,
        force_overwrite: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.split_seed = split_seed
        self.row_shuffle_seed = row_shuffle_seed
        self.train_chunk_size = train_chunk_size
        self.test_chunk_size = test_chunk_size
        self.test_size = test_size
        self.shuffle_columns = shuffle_columns
        self.max_workers_param = max_workers_param
        self.force_overwrite = force_overwrite

    def run(self):
        """Public API: run the dataset split & chunk processing pipeline."""
        return self.process_and_split()

    def process_and_split(self):
        """Main method that orchestrates the dataset splitting and chunk processing.

        This largely mirrors the original script's `process_and_split` function
        but uses instance state and respects the new max_workers_param policy.
        """
        np.random.seed(self.split_seed)

        suffix = f"{self.dataset_name}_Sseed{self.split_seed}_trainSize{self.train_chunk_size}_testSize{self.test_chunk_size}"

        # Ensure base output directory exists (create if missing)
        os.makedirs(self.output_dir, exist_ok=True)

        dataset_output_dir = os.path.join(self.output_dir, self.dataset_name, suffix)
        rseed_output_dir = os.path.join(dataset_output_dir, f"Rseed{self.row_shuffle_seed}")

        # If the target Rseed directory already exists and is not empty, offer to delete it
        if os.path.exists(rseed_output_dir) and os.listdir(rseed_output_dir):
            # Prepare multi-layer framed warning for prominence
            msg_lines = [
                "ERROR: Target Rseed output directory already exists and is not empty.",
                "To avoid accidental overwrite, you can choose to delete and recreate it.",
                f"Target: {rseed_output_dir}",
                "Do you want to delete this directory and recreate it? (Y/N)",
            ]

            # Compute a comfortable width for the frames
            inner_width = max(80, max(len(s) for s in msg_lines) + 6)
            outer_width = inner_width + 6

            # Print nested frames
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

            # Decide choice based on force flag or non-interactive environment
            interactive_mode = sys.stdin.isatty()
            if getattr(self, 'force_overwrite', False) or (not interactive_mode):
                choice = 'y'
                if getattr(self, 'force_overwrite', False):
                    print("--force_overwrite flag detected: proceeding to delete and recreate the directory.")
                else:
                    print("Non-interactive environment detected: defaulting to delete and recreate (no prompt).")
            else:
                try:
                    choice = input("Delete and recreate the existing directory? [y/N]: ").strip().lower()
                except Exception:
                    print("No input available or input interrupted. Aborting.")
                    sys.exit(1)

            if choice in ("y", "yes"):
                # Attempt to remove the Rseed directory tree
                try:
                    shutil.rmtree(rseed_output_dir)
                    print(f"Removed existing Rseed directory: {rseed_output_dir}")
                except Exception as exc:
                    print(f"Failed to remove Rseed directory {rseed_output_dir}: {exc}")
                    sys.exit(1)

                # Recreate the empty subdirectories with Rseed layer
                rseed_dir = os.path.join(dataset_output_dir, f"Rseed{self.row_shuffle_seed}")
                os.makedirs(os.path.join(rseed_dir, "X_train"), exist_ok=True)
                os.makedirs(os.path.join(rseed_dir, "y_train"), exist_ok=True)
                os.makedirs(os.path.join(rseed_dir, "X_test"), exist_ok=True)
                os.makedirs(os.path.join(rseed_dir, "y_test"), exist_ok=True)

                print(f"Recreated empty Rseed directory structure under: {rseed_output_dir}")
            else:
                print("Operation aborted by user. Exiting without changes.")
                sys.exit(0)

        # Build expected subdirectories under dataset_output_dir/Rseed{row_shuffle_seed} if not present
        rseed_dir = os.path.join(dataset_output_dir, f"Rseed{self.row_shuffle_seed}")
        os.makedirs(os.path.join(rseed_dir, "X_train"), exist_ok=True)
        os.makedirs(os.path.join(rseed_dir, "y_train"), exist_ok=True)
        os.makedirs(os.path.join(rseed_dir, "X_test"), exist_ok=True)
        os.makedirs(os.path.join(rseed_dir, "y_test"), exist_ok=True)

        input_file = os.path.join(self.input_dir, f"{self.dataset_name}.csv")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        print(f"[Info] Reading dataset: {input_file}")
        df = pd.read_csv(input_file, encoding="utf-8")

        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.split_seed)

        # Analyze labels at the dataset level for consistent transformation
        all_original_labels = df.iloc[:, -1]  # Get all labels from the original dataset

        # Use dataset-level helper to prepare label mapping and pseudo label
        label_mapping, mapping_path = build_dataset_context(
            all_original_labels, dataset_output_dir
        )
        if mapping_path is not None:
            print(f"[Info] Saved label mapping info to: {mapping_path}")

        # Process test data in chunks directly from memory (no need to save and reload)
        test_chunks = [test_df[i:i + self.test_chunk_size] for i in range(0, len(test_df), self.test_chunk_size)]

        # Pre-generate all training chunks to avoid repeated sampling (performance optimization)
        print(f"[Optimization] Pre-generating {len(test_chunks)} training chunks...")
        train_chunks = []
        for i in range(len(test_chunks)):
            if self.train_chunk_size <= len(train_df):
                train_chunk = train_df.sample(n=self.train_chunk_size, random_state=self.split_seed + i)
            else:
                train_chunk = train_df.copy()
            train_chunk = train_chunk.sample(frac=1, random_state=self.row_shuffle_seed).reset_index(drop=True)
            train_chunks.append(train_chunk)

        print(f"[Optimization] Pre-generation completed. Starting parallel chunk processing...")

        # Prepare arguments for parallel processing
        chunk_args = []
        for i, (test_chunk, train_chunk) in enumerate(zip(test_chunks, train_chunks)):
            # order: i, dataset_output_dir, label_mapping, row_shuffle_seed, shuffle_columns, test_chunk, train_chunk
            args = (i, dataset_output_dir, label_mapping,
                    self.row_shuffle_seed, self.shuffle_columns,
                    test_chunk, train_chunk)
            chunk_args.append(args)

        # Determine optimal number of workers based on policy
        if self.max_workers_param == 1:
            # Serial processing
            print(f"[Optimization] Running serial processing over {len(chunk_args)} chunks...")
            completed = 0
            with tqdm(total=len(chunk_args),
                      desc=f"Processing {self.dataset_name} chunks (seed={self.row_shuffle_seed}, train_size={self.train_chunk_size})") as pbar:
                for args in chunk_args:
                    process_single_chunk(args)
                    completed += 1
                    pbar.update(1)
                    pbar.set_postfix({'completed': completed})

            print(f"[Success] All {len(chunk_args)} chunks processed successfully (serial)!")
            return

        # Parallel processing
        cpu_count = multiprocessing.cpu_count()
        if self.max_workers_param == -1:
            max_workers = min(cpu_count, len(chunk_args))
        else:
            max_workers = min(cpu_count, len(chunk_args), self.max_workers_param)

        # Ensure at least one worker (good fallback if there are zero chunks)
        max_workers = max(1, max_workers)

        print(f"[Optimization] Using {max_workers} parallel workers for {len(chunk_args)} chunks...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(process_single_chunk, args): args[0] for args in chunk_args}

            completed = 0
            with tqdm(total=len(chunk_args),
                      desc=f"Processing {self.dataset_name} chunks (seed={self.row_shuffle_seed}, train_size={self.train_chunk_size})") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_idx, result = future.result()
                    completed += 1
                    pbar.update(1)
                    pbar.set_postfix({'completed': completed})

        print(f"[Success] All {len(chunk_args)} chunks processed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Folder containing the dataset CSV")
    parser.add_argument("--output_dir", required=True,
                        help="Base output directory")
    
    parser.add_argument("--dataset_name", required=True,
                        help="Name of the dataset (e.g., bank)")

    parser.add_argument("--split_seed", type=int, default=42,
                       help="Seed used for dataset-level splitting and deterministic sampling (default: 42)")
    parser.add_argument("--row_shuffle_seed", type=int, default=123,
                        help="Used to shuffle rows inside training chunks (per-chunk seed)")

    parser.add_argument("--train_chunk_size", type=int, default=32,
                        help="Number of few-shot examples (shots) included in each prompt for training chunks; used as the few-shot sample count per prompt")
    parser.add_argument("--test_chunk_size", type=int, default=7,
                        help="Number of items aggregated into a single prompt for LLM inference (per-prompt inference batch size)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of the dataset reserved for testing (proportion used to split train/test)")

    parser.add_argument("--shuffle_columns", type=lambda v: v.lower() in ("true", "1", "yes", "y"), default=False,
                       help="Enable column shuffling based on per-chunk seed (True/False) (default: False)")

    parser.add_argument("--max_workers", type=int, default=64,
                        help="Max workers parameter. If set to 1, processing will run serially. If -1, use min(cpu_count, n_chunks). Otherwise use min(cpu_count, n_chunks, value). Default=64 (preserves previous cap)")

    parser.add_argument("--force_overwrite", action="store_true", default=False,
                        help="If set, delete existing dataset output directory without prompting. Note: when running interactively the script defaults to delete-and-recreate even without this flag.")



    args = parser.parse_args()

    print(f"[DataPrep] Starting data preparation with modular extensions...")
    print(f"[DataPrep] Output will be written to local path for fast IO")

    runner = DataChunkPrepRunner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,

        split_seed=args.split_seed,
        row_shuffle_seed=args.row_shuffle_seed,

        train_chunk_size=args.train_chunk_size,
        test_chunk_size=args.test_chunk_size,
        test_size=args.test_size,

        shuffle_columns=args.shuffle_columns,
        max_workers_param=args.max_workers,
        
        force_overwrite=args.force_overwrite
    )

    runner.run()

    print(f"[DataPrep] Data preparation completed successfully!")
