#!/usr/bin/env python3
"""
Layer 1: File Discovery Layer

Responsibilities:
1. Parse input_dir parameter
2. If input is a .jsonl file: directly return [file path]
3. If input is a directory: automatically discover target files based on other parameters
4. Return: List[str] - List of file paths to be evaluated

Input: args (command line arguments)
Output: List[str] - List of file paths
"""

import os
import glob
from typing import List


class FileDiscoveryLayer:
    
    @staticmethod
    def discover_files(args) -> List[str]:
        """
        Discover files to be evaluated
        
        Args:
            args: Command line arguments object
            
        Returns:
            List[str]: List of file paths
        """
        input_dir = args.input_dir
        
        if input_dir.endswith('.jsonl'):
            return FileDiscoveryLayer._handle_single_file(input_dir)
        
        return FileDiscoveryLayer._handle_directory(input_dir, args)
    
    @staticmethod
    def _handle_single_file(file_path: str) -> List[str]:
        """
        Handle single file input
        
        Args:
            file_path: Path to .jsonl file
            
        Returns:
            List[str]: List containing a single file path
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")
            
        print(f"🔍 Single file mode: {file_path}")
        return [file_path]
    
    @staticmethod
    def _handle_directory(dir_path: str, args) -> List[str]:
        """
        Handle directory input, discover target files based on parameters
        
        Args:
            dir_path: Directory path
            args: Command line arguments object
            
        Returns:
            List[str]: List of discovered file paths
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Path is not a directory: {dir_path}")
        
        print(f"🔍 Directory mode: {dir_path}")
        
        required_params = ['train_chunk_size', 'test_chunk_size']
        missing_params = [param for param in required_params 
                         if not hasattr(args, param) or getattr(args, param) is None]
        
        if missing_params:
            raise ValueError(f"Directory mode missing required parameters: {missing_params}")
        
        found_files = FileDiscoveryLayer._find_matching_files(dir_path, args)
        
        if not found_files:
            raise FileNotFoundError(f"No matching files found in directory {dir_path}")
        
        print(f"📁 Found {len(found_files)} matching files")
        return found_files
    
    @staticmethod
    def _find_matching_files(dir_path: str, args) -> List[str]:
        """
        Find matching files in directory
        
        Args:
            dir_path: Search directory
            args: Arguments object
            
        Returns:
            List[str]: List of matching file paths
        """
        dataset_name = getattr(args, 'dataset_name', None)
        model_name = getattr(args, 'model_name', None)
        split_seed = getattr(args, 'split_seed', None)
        row_shuffle_seeds = getattr(args, 'row_shuffle_seeds', None)
        train_chunk_size = getattr(args, 'train_chunk_size', None)
        test_chunk_size = getattr(args, 'test_chunk_size', None)
        
        base_model_name = None
        if model_name:
            if '::' in model_name:
                parts = model_name.split('::', 1)
                backend, actual_model = parts[0], parts[1]
                if backend.lower() == 'openai':
                    model_name = actual_model
                else:
                    model_name = model_name.replace('::', '_')
            
            # Handle HuggingFace format paths (e.g., minzl/toy_3550)
            if '/' in model_name:
                model_name = model_name.split('/')[-1]
            
            base_model_name = model_name.replace('-', '_').replace('.', '_')
        
        print(f"🔍 Search parameters:")
        print(f"   dataset_name: {dataset_name}")
        print(f"   model_name: {base_model_name}")
        print(f"   split_seed: {split_seed}")
        print(f"   row_shuffle_seeds: {row_shuffle_seeds}")
        print(f"   train_chunk_size: {train_chunk_size}")
        print(f"   test_chunk_size: {test_chunk_size}")
        
        found_files = []
        
        base_pattern = "*.jsonl"
        all_files = glob.glob(os.path.join(dir_path, "**", base_pattern), recursive=True)
        
        print(f"📁 Found {len(all_files)} .jsonl files, starting filtering...")
        
        for file_path in all_files:
            if FileDiscoveryLayer._matches_criteria(file_path, dataset_name, base_model_name, 
                                                   split_seed, row_shuffle_seeds, 
                                                   train_chunk_size, test_chunk_size):
                found_files.append(file_path)
        
        found_files = sorted(list(set(found_files)))
        
        print(f"📁 Found {len(found_files)} matching files after filtering")
        return found_files
    
    @staticmethod
    def _matches_criteria(file_path: str, dataset_name=None, model_name=None, 
                         split_seed=None, row_shuffle_seeds=None, 
                         train_chunk_size=None, test_chunk_size=None) -> bool:
        """
        Check if file matches specified criteria
        
        Args:
            file_path: File path
            dataset_name: Dataset name (optional, None means fuzzy match)
            model_name: Model name (optional, None means fuzzy match)
            split_seed: Split seed (optional, None means fuzzy match)
            row_shuffle_seeds: Row shuffle seed list (optional, None means fuzzy match)
            train_chunk_size: Training chunk size (required)
            test_chunk_size: Test chunk size (required)
            
        Returns:
            bool: Whether it matches
        """
        filename = os.path.basename(file_path)
        file_parts = file_path.split(os.sep)
        
        if dataset_name:
            if dataset_name not in filename and dataset_name not in ' '.join(file_parts):
                return False
        
        if model_name:
            if not FileDiscoveryLayer._is_exact_model_match(filename, model_name):
                return False
        
        if split_seed is not None:
            if f"Sseed{split_seed}" not in ' '.join(file_parts):
                return False
        
        if row_shuffle_seeds:
            has_matching_seed = False
            for seed in row_shuffle_seeds:
                if f"Rseed{seed}" in filename:
                    has_matching_seed = True
                    break
            if not has_matching_seed:
                return False
        
        if train_chunk_size is not None:
            if f"trainSize{train_chunk_size}" not in ' '.join(file_parts):
                return False
        
        if test_chunk_size is not None:
            if f"testSize{test_chunk_size}" not in ' '.join(file_parts):
                return False
        
        return True
    
    @staticmethod
    def _is_exact_model_match(filename: str, model_name: str) -> bool:
        """
        Check if filename exactly matches model name
        
        Args:
            filename: Filename
            model_name: Model name
            
        Returns:
            bool: Whether it's an exact match
        """
        
        if '@@' in filename:
            # Extract model part (before @@)
            model_part = filename.split('@@')[0]
            
            
            if model_part == model_name:
                return True
            
            if model_part.startswith(model_name + '_'):
                return False
                
            return model_part == model_name
        else:
            return model_name in filename and not filename.replace(model_name, '').startswith('_')
    
    @staticmethod
    def extract_dataset_name_from_file(file_path: str) -> str:
        """
        Extract dataset name from file path
        
        Args:
            file_path: File path
            
        Returns:
            str: Dataset name
        """
        filename = os.path.basename(file_path)
        
        # Try to extract dataset name from filename
        if '@@' in filename:
            parts = filename.split('@@')
            if len(parts) >= 2:
                # Extract from second part: bank_Rseed41_trainSize32_testSize7.jsonl -> bank
                second_part = parts[1]
                if '_' in second_part:
                    return second_part.split('_')[0]
        
        if '_' in filename:
            return filename.split('_')[0]
        
        return filename.replace('.jsonl', '')