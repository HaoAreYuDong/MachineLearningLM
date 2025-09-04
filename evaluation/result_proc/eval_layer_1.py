#!/usr/bin/env python3
"""
Layer 1: 文件发现层

职责：
1. 解析input_dir参数
2. 如果是.jsonl文件：直接返回[文件路径]
3. 如果是目录：根据其他参数自动发现目标文件
4. 返回：List[str] - 待评估的文件路径列表

输入：args（命令行参数）
输出：List[str] - 文件路径列表
"""

import os
import glob
from typing import List


class FileDiscoveryLayer:
    """文件发现层 - 负责根据参数发现目标评估文件"""
    
    @staticmethod
    def discover_files(args) -> List[str]:
        """
        发现待评估的文件列表
        
        Args:
            args: 命令行参数对象
            
        Returns:
            List[str]: 文件路径列表
        """
        input_dir = args.input_dir
        
        # 情况1: input_dir是单个.jsonl文件
        if input_dir.endswith('.jsonl'):
            return FileDiscoveryLayer._handle_single_file(input_dir)
        
        # 情况2: input_dir是目录，需要根据其他参数发现文件
        return FileDiscoveryLayer._handle_directory(input_dir, args)
    
    @staticmethod
    def _handle_single_file(file_path: str) -> List[str]:
        """
        处理单文件输入
        
        Args:
            file_path: .jsonl文件路径
            
        Returns:
            List[str]: 包含单个文件路径的列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")
            
        if not os.path.isfile(file_path):
            raise ValueError(f"路径不是文件: {file_path}")
            
        print(f"SEARCH: 单文件模式: {file_path}")
        return [file_path]
    
    @staticmethod
    def _handle_directory(dir_path: str, args) -> List[str]:
        """
        处理目录输入，根据参数发现目标文件
        
        Args:
            dir_path: 目录路径
            args: 命令行参数对象
            
        Returns:
            List[str]: 发现的文件路径列表
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"路径不是目录: {dir_path}")
        
        print(f"SEARCH: 目录模式: {dir_path}")
        
        # 验证目录模式的必需参数
        required_params = ['train_chunk_size', 'test_chunk_size']
        missing_params = [param for param in required_params 
                         if not hasattr(args, param) or getattr(args, param) is None]
        
        if missing_params:
            raise ValueError(f"目录模式缺少必需参数: {missing_params}")
        
        # 根据参数构建文件匹配模式
        found_files = FileDiscoveryLayer._find_matching_files(dir_path, args)
        
        if not found_files:
            raise FileNotFoundError(f"在目录 {dir_path} 中未找到匹配的文件")
        
        print(f"OUTPUT: 发现 {len(found_files)} 个匹配文件")
        return found_files
    
    @staticmethod
    def _find_matching_files(dir_path: str, args) -> List[str]:
        """
        在目录中查找匹配的文件
        
        Args:
            dir_path: 搜索目录
            args: 参数对象
            
        Returns:
            List[str]: 匹配的文件路径列表
        """
        # 获取参数（可选参数可能为None）
        dataset_name = getattr(args, 'dataset_name', None)
        model_name = getattr(args, 'model_name', None)
        split_seed = getattr(args, 'split_seed', None)
        row_shuffle_seeds = getattr(args, 'row_shuffle_seeds', None)
        train_chunk_size = getattr(args, 'train_chunk_size', None)
        test_chunk_size = getattr(args, 'test_chunk_size', None)
        
        # 处理模型名称，提取基础名称（去掉路径前缀）
        base_model_name = None
        if model_name:
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
            
            # 替换其他可能的特殊字符为下划线
            base_model_name = model_name.replace('-', '_').replace('.', '_')
        
        print(f"SEARCH: 搜索参数:")
        print(f"   dataset_name: {dataset_name}")
        print(f"   model_name: {base_model_name}")
        print(f"   split_seed: {split_seed}")
        print(f"   row_shuffle_seeds: {row_shuffle_seeds}")
        print(f"   train_chunk_size: {train_chunk_size}")
        print(f"   test_chunk_size: {test_chunk_size}")
        
        # 构建文件匹配模式
        found_files = []
        
        # 基础搜索模式
        base_pattern = "*.jsonl"
        all_files = glob.glob(os.path.join(dir_path, "**", base_pattern), recursive=True)
        
        print(f"OUTPUT: 找到 {len(all_files)} 个 .jsonl 文件，开始过滤...")
        
        # 过滤文件
        for file_path in all_files:
            if FileDiscoveryLayer._matches_criteria(file_path, dataset_name, base_model_name, 
                                                   split_seed, row_shuffle_seeds, 
                                                   train_chunk_size, test_chunk_size):
                found_files.append(file_path)
        
        # 去重并排序
        found_files = sorted(list(set(found_files)))
        
        print(f"OUTPUT: 过滤后发现 {len(found_files)} 个匹配文件")
        return found_files
    
    @staticmethod
    def _matches_criteria(file_path: str, dataset_name=None, model_name=None, 
                         split_seed=None, row_shuffle_seeds=None, 
                         train_chunk_size=None, test_chunk_size=None) -> bool:
        """
        检查文件是否匹配指定条件
        
        Args:
            file_path: 文件路径
            dataset_name: 数据集名称（可选，None表示模糊匹配）
            model_name: 模型名称（可选，None表示模糊匹配）
            split_seed: 分割种子（可选，None表示模糊匹配）
            row_shuffle_seeds: 行随机种子列表（可选，None表示模糊匹配）
            train_chunk_size: 训练块大小（必需）
            test_chunk_size: 测试块大小（必需）
            
        Returns:
            bool: 是否匹配
        """
        filename = os.path.basename(file_path)
        file_parts = file_path.split(os.sep)
        
        # 1. 检查数据集名称（如果指定）
        if dataset_name:
            if dataset_name not in filename and dataset_name not in ' '.join(file_parts):
                return False
        
        # 2. 检查模型名称（如果指定，使用精确匹配）
        if model_name:
            if not FileDiscoveryLayer._is_exact_model_match(filename, model_name):
                return False
        
        # 3. 检查分割种子（如果指定）
        if split_seed is not None:
            # 在路径中查找 Sseed{split_seed}
            if f"Sseed{split_seed}" not in ' '.join(file_parts):
                return False
        
        # 4. 检查行随机种子（如果指定）
        if row_shuffle_seeds:
            # 检查是否包含任一指定的行随机种子
            has_matching_seed = False
            for seed in row_shuffle_seeds:
                if f"Rseed{seed}" in filename:
                    has_matching_seed = True
                    break
            if not has_matching_seed:
                return False
        
        # 5. 检查训练块大小（必需）
        if train_chunk_size is not None:
            if f"trainSize{train_chunk_size}" not in ' '.join(file_parts):
                return False
        
        # 6. 检查测试块大小（必需）
        if test_chunk_size is not None:
            if f"testSize{test_chunk_size}" not in ' '.join(file_parts):
                return False
        
        return True
    
    @staticmethod
    def _is_exact_model_match(filename: str, model_name: str) -> bool:
        """
        检查文件名是否精确匹配模型名称
        
        Args:
            filename: 文件名
            model_name: 模型名称
            
        Returns:
            bool: 是否精确匹配
        """
        # 分析文件名格式，通常是: toy_3550@@bank_Rseed40_trainSize32_testSize7.jsonl
        # 或者: toy_3550_1565_1700@@bank_Rseed40_trainSize32_testSize7.jsonl
        
        if '@@' in filename:
            # 提取模型部分（@@之前的部分）
            model_part = filename.split('@@')[0]
            
            # 精确匹配：模型名称应该完全相同，或者以特定分隔符结束
            # 例如: toy_3550 应该匹配 toy_3550，但不应该匹配 toy_3550_1565_1700
            
            # 情况1：完全匹配
            if model_part == model_name:
                return True
            
            # 情况2：模型名称后面跟着下划线和其他内容（避免部分匹配）
            # toy_3550 不应该匹配 toy_3550_1565_1700
            if model_part.startswith(model_name + '_'):
                return False
                
            # 情况3：检查是否是真正的精确匹配
            return model_part == model_name
        else:
            # 没有::的情况，直接检查文件名
            return model_name in filename and not filename.replace(model_name, '').startswith('_')
    
    @staticmethod
    def extract_dataset_name_from_file(file_path: str) -> str:
        """
        从文件路径提取数据集名称
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 数据集名称
        """
        filename = os.path.basename(file_path)
        
        # 尝试从文件名中提取数据集名称
        if '@@' in filename:
            # 格式: toy_3550@@bank_Rseed41_trainSize32_testSize7.jsonl
            parts = filename.split('@@')
            if len(parts) >= 2:
                # 从第二部分提取: bank_Rseed41_trainSize32_testSize7.jsonl -> bank
                second_part = parts[1]
                if '_' in second_part:
                    return second_part.split('_')[0]
        
        # 备选方案：直接从文件名提取
        if '_' in filename:
            return filename.split('_')[0]
        
        # 最后方案：去掉扩展名
        return filename.replace('.jsonl', '')
