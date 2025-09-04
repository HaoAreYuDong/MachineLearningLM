#!/usr/bin/env python3
"""
Layer 2: 标签统计层

职责：
1. 接收文件路径列表
2. 扫描所有文件的ground truth标签
3. 统计出现频率最高的标签作为默认标签
4. 返回每个文件对应的默认标签

输入：List[str] - 文件路径列表
输出：Dict[str, int] - {文件名: 默认标签}
"""
import re
import os
import json
from typing import List, Dict
from collections import Counter


class LabelStatisticsLayer:
    """标签统计层 - 负责统计ground truth分布，确定默认标签"""
    
    @staticmethod
    def compute_default_labels(file_paths: List[str]) -> Dict[str, int]:
        """
        计算每个文件的默认标签
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            Dict[str, int]: {文件路径: 默认标签}
        """
        print("INFO: Layer 2: 标签统计...")
        
        # 全局标签统计（用于所有文件的默认标签）
        global_label_counts = Counter()
        file_label_info = {}
        
        for file_path in file_paths:
            print(f"📈 统计标签: {os.path.basename(file_path)}")
            
            try:
                file_labels = LabelStatisticsLayer._extract_labels_from_file(file_path)
                
                # 更新全局统计
                global_label_counts.update(file_labels)
                
                # 记录文件标签信息
                file_label_counts = Counter(file_labels)
                file_label_info[file_path] = {
                    'label_counts': dict(file_label_counts),
                    'total_labels': len(file_labels),
                    'unique_labels': len(file_label_counts)
                }
                
            except Exception as e:
                print(f"WARNING: 统计 {os.path.basename(file_path)} 失败: {e}")
                file_label_info[file_path] = {
                    'error': str(e),
                    'label_counts': {},
                    'total_labels': 0,
                    'unique_labels': 0
                }
        
        # 确定全局默认标签
        if global_label_counts:
            global_default_label = global_label_counts.most_common(1)[0][0]
        else:
            global_default_label = 0  # 如果完全没有标签，使用0
        print(f"TARGET: 确定默认标签: {global_default_label}")
        
        # 为每个文件分配默认标签（目前使用全局默认，后续可优化为文件特定）
        default_labels = {}
        for file_path in file_paths:
            # 可以根据文件特定的标签分布来决定，这里简化为使用全局默认
            file_info = file_label_info[file_path]
            if file_info.get('label_counts'):
                # 使用该文件最频繁的标签，如果没有则使用全局默认
                file_counter = Counter(file_info['label_counts'])
                file_default = file_counter.most_common(1)[0][0] if file_counter else global_default_label
            else:
                file_default = global_default_label
            
            default_labels[file_path] = file_default
            print(f"   OUTPUT: {os.path.basename(file_path)}: 默认标签 = {file_default}")
        
        return {
            'global_label_stats': dict(global_label_counts),
            'default_labels': default_labels,
            'file_label_info': file_label_info
        }
    
    @staticmethod
    def _extract_labels_from_file(file_path: str) -> List[int]:
        """
        从单个文件中提取所有ground truth标签
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[int]: 标签列表
        """
        labels = []
        
        # 支持两种形式：
        # 1) "groundtruth": "[{\"id\": 0, \"label\": 0}, ...]"  (quoted JSON string)
        # 2) "groundtruth": [{"id": 0, "label": 0}, ...]          (direct JSON array)
        quoted_pattern = re.compile(r'"groundtruth"\s*:\s*("(?:(?:\\.|[^"\\])*)")')
        array_pattern = re.compile(r'"groundtruth"\s*:\s*(\[.*?\])')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    gt_parsed = None
                    # 优先匹配带引号的情况："groundtruth": "[...]"
                    m = quoted_pattern.search(line)
                    if m:
                        quoted_val = m.group(1)  # 包含外层双引号
                        try:
                            # 先解码外层的 JSON 字符串，得到内部的 JSON 文本
                            inner_json_text = json.loads(quoted_val)
                            # 再解析内部的 JSON 文本为列表
                            gt_parsed = json.loads(inner_json_text)
                        except Exception:
                            gt_parsed = None
                    else:
                        # 尝试匹配直接的 JSON 数组："groundtruth": [{...}]
                        m2 = array_pattern.search(line)
                        if m2:
                            try:
                                gt_parsed = json.loads(m2.group(1))
                            except Exception:
                                gt_parsed = None
                    
                    if not gt_parsed:
                        continue
                    
                    # 收集标签（期望每个 gt_item 是 dict 并包含 'label' 键）
                    for gt_item in gt_parsed:
                        if isinstance(gt_item, dict) and 'label' in gt_item:
                            labels.append(gt_item['label'])
        except Exception as e:
            raise Exception(f"读取文件失败: {e}")
        
        return labels
