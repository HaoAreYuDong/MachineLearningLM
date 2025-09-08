#!/usr/bin/env python3
"""
Layer 2: Label Statistics Layer

Responsibilities:
1. Receive list of file paths
2. Scan ground truth labels from all files
3. Determine the most frequent label as the default label
4. Return the default label for each file

Input: List[str] - File path list
Output: Dict[str, int] - {Filename: Default label}
"""
import re
import os
import json
from typing import List, Dict
from collections import Counter


class LabelStatisticsLayer:
    """Label Statistics Layer - Responsible for analyzing ground truth distribution and determining default labels"""
    @staticmethod
    def compute_default_labels(file_paths: List[str]) -> Dict[str, int]:
        """
        Compute default label for each file
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dict[str, int]: {File path: Default label}
        """
        print("📊 Layer 2: Label Statistics...")
        
        # Global label statistics (for default label across all files)
        global_label_counts = Counter()
        file_label_info = {}
        
        for file_path in file_paths:
            print(f"📈 Counting labels: {os.path.basename(file_path)}")
            
            try:
                file_labels = LabelStatisticsLayer._extract_labels_from_file(file_path)
                
                global_label_counts.update(file_labels)
                
                file_label_counts = Counter(file_labels)
                file_label_info[file_path] = {
                    'label_counts': dict(file_label_counts),
                    'total_labels': len(file_labels),
                    'unique_labels': len(file_label_counts)
                }
                
            except Exception as e:
                print(f"⚠️ Failed to count labels for {os.path.basename(file_path)}: {e}")
                file_label_info[file_path] = {
                    'error': str(e),
                    'label_counts': {},
                    'total_labels': 0,
                    'unique_labels': 0
                }
        
        if global_label_counts:
            global_default_label = global_label_counts.most_common(1)[0][0]
        else:
            global_default_label = 0  # Use 0 if no labels found
        print(f"🎯 Determined default label: {global_default_label}")
        
        default_labels = {}
        for file_path in file_paths:
            file_info = file_label_info[file_path]
            if file_info.get('label_counts'):
                file_counter = Counter(file_info['label_counts'])
                file_default = file_counter.most_common(1)[0][0] if file_counter else global_default_label
            else:
                file_default = global_default_label
            
            default_labels[file_path] = file_default
            print(f"   📁 {os.path.basename(file_path)}: Default label = {file_default}")
        
        return {
            'global_label_stats': dict(global_label_counts),
            'default_labels': default_labels,
            'file_label_info': file_label_info
        }
    
    @staticmethod
    def _extract_labels_from_file(file_path: str) -> List[int]:
        """
        Extract all ground truth labels from a single file
        
        Args:
            file_path: File path
            
        Returns:
            List[int]: List of labels
        """
        labels = []
        
        # Support two formats:
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
                    # First match quoted case: "groundtruth": "[...]"
                    m = quoted_pattern.search(line)
                    if m:
                        quoted_val = m.group(1)  # Contains outer quotes
                        try:
                            # Decode outer JSON string to get inner JSON text
                            inner_json_text = json.loads(quoted_val)
                            # Parse inner JSON text as list
                            gt_parsed = json.loads(inner_json_text)
                        except Exception:
                            gt_parsed = None
                    else:
                        # Try matching direct JSON array: "groundtruth": [{...}]
                        m2 = array_pattern.search(line)
                        if m2:
                            try:
                                gt_parsed = json.loads(m2.group(1))
                            except Exception:
                                gt_parsed = None
                    
                    if not gt_parsed:
                        continue
                    
                    # Collect labels (expect each gt_item to be dict with 'label' key)
                    for gt_item in gt_parsed:
                        if isinstance(gt_item, dict) and 'label' in gt_item:
                            labels.append(gt_item['label'])
        except Exception as e:
            raise Exception(f"File read failed: {e}")
        
        return labels