#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_layer_3.py - 第三层：单文件评估和标记

处理单个文件的评估逻辑，区分真实预测和默认填充的数据，并为每个预测添加标记。
解析失败时使用高频标签填充，并标记为默认值。

Author: Assistant
Date: 2025-01-20
"""

import json
import pandas as pd
import re
from typing import Dict, List, Tuple, Any, Optional
import traceback
import logging
from sklearn.metrics import classification_report, roc_auc_score

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TaggedSingleFileEvaluationLayer:
    """第三层：单文件评估和标记层
    
    负责处理单个文件的评估逻辑，区分真实预测和默认填充的数据，
    并为每个预测添加标记（'real'表示真实预测，'default'表示默认填充）。
    """
    
    def __init__(self):
        """初始化单文件评估层"""
        pass
    
    @staticmethod
    def evaluate_single_file_with_tags(file_path: str, label_stats: Dict[str, int]) -> Dict[str, Any]:
        """评估单个文件，区分真实预测和默认填充，并添加标记
        
        Args:
            file_path: 要评估的文件路径
            label_stats: 标签统计信息，用于获取高频标签作为默认值
            
        Returns:
            包含真实标签、预测标签、标记和元数据的字典
        """
        try:
            logger.info(f"正在评估单文件: {file_path}")
            
            # 获取默认标签（高频标签）
            if not label_stats:
                default_label = "0"  # 默认为0
            else:
                default_label = str(max(label_stats.items(), key=lambda x: x[1])[0])
            
            logger.info(f"开始评估文件，默认标签设为: {default_label}")
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"文件为空: {file_path}")
                return {
                    'y_true': [],
                    'y_pred': [],
                    'tags': [],
                    'file_path': file_path,
                    'total_samples': 0,
                    'real_predictions': 0,
                    'default_fillings': 0,
                    'default_label': default_label
                }
            
            # 初始化跟踪变量
            y_true = []  # 真实标签
            y_pred = []  # 预测标签  
            y_probs = [] # 正类概率(label=1)
            tags = []    # 解析状态标记
            
            # 统计计数器
            real_count = 0
            default_count = 0
            json_error_count = 0
            
            lines = content.strip().split('\n')
            
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                    
                try:
                    # 主要逻辑：解析JSON
                    data = json.loads(line)
                    line_id = data.get('id', line_num)  # 获取行ID，用于构建样本ID
                    
                    # 解析response和groundtruth
                    try:
                        response_data = json.loads(data['response'])
                        groundtruth_data = json.loads(data['groundtruth'])
                        
                        # 检查长度一致性
                        if len(response_data) != len(groundtruth_data):
                            logger.warning(f"第{line_num}行response和groundtruth长度不匹配，触发异常情况处理")
                            raise ValueError("Response和groundtruth长度不匹配")
                        
                        # 逐个处理每个样本
                        for idx, (pred_item, truth_item) in enumerate(zip(response_data, groundtruth_data)):
                            sample_id = f"L{line_id}R{idx}"  # 构建样本ID: L{line_id}R{response_id}
                            
                            # 提取真实标签
                            true_label = str(truth_item.get('label', default_label))
                            
                            # 提取预测标签
                            pred_result = TaggedSingleFileEvaluationLayer._process_single_prediction(
                                pred_item, data, idx, default_label, sample_id
                            )
                            
                            # 解包结果
                            pred_label = pred_result['pred_label']
                            auc_prob = pred_result['auc_prob'] 
                            tag = pred_result['tag']
                            
                            # 添加到结果列表
                            y_true.append(true_label)
                            y_pred.append(pred_label)
                            y_probs.append(auc_prob)
                            tags.append(tag)
                            
                            # 更新计数器
                            if tag == 'parser_success':
                                real_count += 1
                            else:
                                default_count += 1
                                
                    except Exception as e:
                        # 异常情况一：解析JSON成功，但response处理有错误
                        logger.warning(f"第{line_num}行response处理失败: {e}，调用独立函数恢复")
                        
                        # 调用独立函数a和b处理
                        recovered_samples = TaggedSingleFileEvaluationLayer._handle_response_error(
                            data, line_num, default_label
                        )
                        
                        # 添加恢复的样本
                        for sample_result in recovered_samples:
                            y_true.append(sample_result['true_label'])
                            y_pred.append(sample_result['pred_label'])
                            y_probs.append(sample_result['auc_prob'])
                            tags.append(sample_result['tag'])
                            
                            if sample_result['tag'] == 'parser_success':
                                real_count += 1
                            else:
                                default_count += 1
                        
                        json_error_count += 1
                    
                except Exception as e:
                    # 鲁棒性1：解析JSON失败，调用独立函数a和b处理
                    logger.warning(f"第{line_num}行JSON解析失败: {e}，调用独立函数恢复")
                    
                    recovered_samples = TaggedSingleFileEvaluationLayer._handle_json_parse_error(
                        line, line_num, default_label
                    )
                    
                    # 添加恢复的样本
                    for sample_result in recovered_samples:
                        y_true.append(sample_result['true_label'])
                        y_pred.append(sample_result['pred_label']) 
                        y_probs.append(sample_result['auc_prob'])
                        tags.append(sample_result['tag'])
                        
                        if sample_result['tag'] == 'parser_success':
                            real_count += 1
                        else:
                            default_count += 1
                    
                    json_error_count += 1
            
            total_samples = len(y_true)
            
            # 计算比例
            real_ratio = (real_count / total_samples * 100) if total_samples > 0 else 0
            
            logger.info(f"文件 {file_path} 评估完成:")
            logger.info(f"  总样本数: {total_samples}")
            logger.info(f"  真实预测: {real_count} ({real_ratio:.2f}%)")
            logger.info(f"  默认填充: {default_count}")
            
            # JSON错误信息
            if json_error_count > 0:
                logger.warning(f"  JSON错误: {json_error_count} 行解析失败")
            
            # 默认填充总统计
            if default_count > 0:
                logger.warning(f"  WARNING:  默认填充: {default_count}/{total_samples} 个样本使用了默认标签 '{default_label}'")
            
            # 生成分类报告和AUC
            classification_report_str = None
            auc_score = None
            
            if total_samples > 0:
                try:
                    # 转换为整数标签用于sklearn
                    y_true_int = [int(label) for label in y_true]
                    y_pred_int = [int(label) for label in y_pred]
                    
                    # 生成分类报告
                    unique_labels = sorted(set(y_true_int))
                    classification_report_str = classification_report(
                        y_true_int, y_pred_int,
                        labels=unique_labels,
                        target_names=[f"Class {label}" for label in unique_labels],
                        digits=4,
                        zero_division=0
                    )
                    
                    # 计算AUC（仅适用于二分类）
                    if len(unique_labels) == 2 and 0 in unique_labels and 1 in unique_labels:
                        try:
                            # 正确的AUC计算：使用真实标签和预测概率
                            # y_probs 已经包含了正类(label=1)的概率
                            auc_score = roc_auc_score(y_true_int, y_probs)
                            logger.info(f"AUC Score: {auc_score:.4f}")
                        except Exception as e:
                            logger.warning(f"AUC计算失败: {e}")
                            # 降级方案：使用预测标签（虽然不准确，但至少有个数值）
                            try:
                                auc_score = roc_auc_score(y_true_int, y_pred_int)
                                logger.warning(f"使用预测标签计算的AUC (不准确): {auc_score:.4f}")
                            except:
                                auc_score = None
                except Exception as e:
                    logger.warning(f"分类报告生成失败: {e}")
            
            return {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_probs': y_probs,  # 添加概率信息
                'tags': tags,
                'file_path': file_path,
                'total_samples': total_samples,
                'real_predictions': real_count,
                'json_errors': json_error_count,
                'default_fillings': default_count,
                'default_label': default_label,
                'classification_report': classification_report_str,
                'auc_score': auc_score
            }
            
        except Exception as e:
            logger.error(f"评估文件 {file_path} 时发生错误: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return {
                'y_true': [],
                'y_pred': [],
                'tags': [],
                'file_path': file_path,
                'total_samples': 0,
                'real_predictions': 0,
                'default_fillings': 0,
                'default_label': label_stats.get('0', '0') if label_stats else '0',
                'error': str(e)
            }
    
    @staticmethod
    def evaluate_files(file_paths: List[str], label_stats: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
        """批量评估多个文件，每个文件都添加标记
        
        Args:
            file_paths: 要评估的文件路径列表
            label_stats: 标签统计信息
            
        Returns:
            每个文件的评估结果字典 {文件路径: 评估结果}
        """
        results = {}
        
        for file_path in file_paths:
            result = TaggedSingleFileEvaluationLayer.evaluate_single_file_with_tags(file_path, label_stats)
            results[file_path] = result
        
        return results
    
    @staticmethod
    def _process_single_prediction(pred_item: dict, data: dict, idx: int, default_label: str, sample_id: str) -> dict:
        """
        处理单个预测项
        
        Args:
            pred_item: response中的单个预测项
            data: 完整的行数据(包含batch_probabilities)
            idx: 样本索引
            default_label: 默认标签
            sample_id: 样本ID
            
        Returns:
            dict: 包含id, pred_label, auc_prob, tag的字典
        """
        # 尝试直接从label字段获取预测
        if 'label' in pred_item:
            try:
                # 验证label是否为有效数字
                pred_label = str(int(pred_item['label']))
                
                # 获取正类概率
                auc_prob = TaggedSingleFileEvaluationLayer._extract_positive_class_prob(data, idx)
                
                return {
                    'id': sample_id,
                    'pred_label': pred_label,
                    'auc_prob': auc_prob,
                    'tag': 'parser_success'
                }
            except (ValueError, TypeError):
                # label不是有效数字，尝试从概率推断
                pass
        
        # label字段无效或不存在，调用独立函数b
        result = TaggedSingleFileEvaluationLayer._extract_prediction_from_probabilities(
            data, idx, default_label, sample_id
        )
        return result
    
    @staticmethod
    def _extract_positive_class_prob(data: dict, idx: int) -> float:
        """
        提取正类(label=1)概率
        
        Args:
            data: 完整行数据
            idx: 样本索引
            
        Returns:
            float: 正类概率，默认0.5
        """
        try:
            batch_probs = data.get('batch_probabilities', [])
            if not isinstance(batch_probs, list) or idx >= len(batch_probs):
                return 0.5
            
            prob_item = batch_probs[idx]
            if not isinstance(prob_item, dict) or 'label_probs' not in prob_item:
                return 0.5
            
            label_probs = prob_item['label_probs']
            if not isinstance(label_probs, list):
                return 0.5
            
            # 构建概率字典
            prob_dict = {}
            for lp in label_probs:
                if isinstance(lp, dict) and 'label' in lp and 'prob' in lp:
                    try:
                        label = str(int(lp['label']))  # 验证是数字标签
                        prob = float(lp['prob'])
                        prob_dict[label] = prob
                    except (ValueError, TypeError):
                        continue
            
            # 返回正类(label=1)概率
            if '1' in prob_dict:
                return prob_dict['1']
            elif '0' in prob_dict:
                return 1.0 - prob_dict['0']  # 二分类情况下，1-负类概率
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    @staticmethod
    def _handle_response_error(data: dict, line_num: int, default_label: str) -> List[dict]:
        """
        异常情况一：解析JSON成功，但response处理有错误
        调用独立函数a和b处理
        
        Args:
            data: 已解析的JSON数据
            line_num: 行号
            default_label: 默认标签
            
        Returns:
            List[dict]: 恢复的样本列表
        """
        # 调用独立函数a：通过正则获取ground_truth
        groundtruth_data = TaggedSingleFileEvaluationLayer._extract_groundtruth_by_regex(data, line_num)
        
        if not groundtruth_data:
            # 如果连groundtruth都获取不到，返回单个默认样本
            return [{
                'id': f"L{line_num}R0",
                'true_label': default_label,
                'pred_label': default_label,
                'auc_prob': 0.5,
                'tag': 'parser_false'
            }]
        
        # 调用独立函数b：根据groundtruth和batch_probabilities恢复预测
        recovered_samples = []
        line_id = data.get('id', line_num)
        
        for idx, truth_item in enumerate(groundtruth_data):
            sample_id = f"L{line_id}R{idx}"
            true_label = str(truth_item.get('label', default_label))
            
            # 调用独立函数b处理每个样本
            pred_result = TaggedSingleFileEvaluationLayer._extract_prediction_from_probabilities(
                data, idx, default_label, sample_id
            )
            
            recovered_samples.append({
                'id': sample_id,
                'true_label': true_label,
                'pred_label': pred_result['pred_label'],
                'auc_prob': pred_result['auc_prob'],
                'tag': pred_result['tag']
            })
        
        return recovered_samples
    
    @staticmethod
    def _handle_json_parse_error(line: str, line_num: int, default_label: str) -> List[dict]:
        """
        鲁棒性1：解析JSON失败，调用独立函数a和b处理
        
        Args:
            line: 原始行数据
            line_num: 行号
            default_label: 默认标签
            
        Returns:
            List[dict]: 恢复的样本列表
        """
        # 调用独立函数a：通过正则获取ground_truth
        groundtruth_data = TaggedSingleFileEvaluationLayer._extract_groundtruth_by_regex_from_line(line)
        
        if not groundtruth_data:
            # 如果连groundtruth都获取不到，返回单个默认样本
            return [{
                'id': f"L{line_num}R0",
                'true_label': default_label,
                'pred_label': default_label,
                'auc_prob': 0.5,
                'tag': 'parser_false'
            }]
        
        # 尝试提取batch_probabilities
        batch_probs_data = TaggedSingleFileEvaluationLayer._extract_batch_probabilities_by_regex(line)
        
        # 构建伪data对象用于调用独立函数b
        fake_data = {
            'id': line_num,
            'batch_probabilities': batch_probs_data
        }
        
        # 调用独立函数b：根据groundtruth恢复预测
        recovered_samples = []
        
        for idx, truth_item in enumerate(groundtruth_data):
            sample_id = f"L{line_num}R{idx}"
            true_label = str(truth_item.get('label', default_label))
            
            # 调用独立函数b处理每个样本
            pred_result = TaggedSingleFileEvaluationLayer._extract_prediction_from_probabilities(
                fake_data, idx, default_label, sample_id
            )
            
            recovered_samples.append({
                'id': sample_id,
                'true_label': true_label,
                'pred_label': pred_result['pred_label'],
                'auc_prob': pred_result['auc_prob'],
                'tag': pred_result['tag']
            })
        
        return recovered_samples
    
    @staticmethod
    def _extract_groundtruth_by_regex(data: dict, line_num: int) -> List[dict]:
        """
        独立函数a：通过正则从已解析数据中获取ground_truth
        (这个百分比是能成功的)
        
        Args:
            data: 已解析的JSON数据
            line_num: 行号(用于日志)
            
        Returns:
            List[dict]: groundtruth数据列表
        """
        try:
            # 先尝试直接获取
            if 'groundtruth' in data:
                groundtruth = data['groundtruth']
                if isinstance(groundtruth, str):
                    return json.loads(groundtruth)
                elif isinstance(groundtruth, list):
                    return groundtruth
            
            logger.warning(f"第{line_num}行无法从data中获取groundtruth")
            return []
            
        except Exception as e:
            logger.warning(f"第{line_num}行groundtruth解析失败: {e}")
            return []
    
    @staticmethod
    def _extract_groundtruth_by_regex_from_line(line: str) -> List[dict]:
        """
        独立函数a：通过正则从原始行中获取ground_truth
        
        Args:
            line: 原始行数据
            
        Returns:
            List[dict]: groundtruth数据列表
        """
        
        try:
            # 使用正则表达式提取groundtruth字段
            groundtruth_pattern = r'"groundtruth":\s*"(\[.*?\])"'
            match = re.search(groundtruth_pattern, line)
            
            if match:
                groundtruth_str = match.group(1).replace('\\"', '"')
                return json.loads(groundtruth_str)
            
            # 尝试另一种格式 "groundtruth": [...]
            groundtruth_pattern2 = r'"groundtruth":\s*(\[.*?\])'
            match2 = re.search(groundtruth_pattern2, line, re.DOTALL)
            
            if match2:
                groundtruth_str = match2.group(1)
                return json.loads(groundtruth_str)
            
            return []
            
        except Exception as e:
            logger.debug(f"正则提取groundtruth失败: {e}")
            return []
    
    @staticmethod
    def _extract_batch_probabilities_by_regex(line: str) -> List[dict]:
        """
        通过正则从原始行中提取batch_probabilities
        
        Args:
            line: 原始行数据
            
        Returns:
            List[dict]: batch_probabilities数据列表
        """
        
        try:
            # 使用正则表达式提取batch_probabilities字段
            pattern = r'"batch_probabilities":\s*(\[.*?\])(?=,\s*"[^"]+"|$)'
            match = re.search(pattern, line, re.DOTALL)
            
            if match:
                batch_probs_str = match.group(1)
                return json.loads(batch_probs_str)
            
            return []
            
        except Exception as e:
            logger.debug(f"正则提取batch_probabilities失败: {e}")
            return []
    
    @staticmethod
    def _extract_prediction_from_probabilities(data: dict, idx: int, default_label: str, sample_id: str) -> dict:
        """
        独立函数b：根据传入的ground_truth和可选参数id，通过batch_probabilities解析预测
        
        Args:
            data: 包含batch_probabilities的数据
            idx: 样本索引(可选参数)
            default_label: 默认标签
            sample_id: 样本ID
            
        Returns:
            dict: 包含id, pred_label, auc_prob, tag的字典
        """
        try:
            batch_probs = data.get('batch_probabilities', [])
            if not isinstance(batch_probs, list):
                return {
                    'id': sample_id,
                    'pred_label': default_label,
                    'auc_prob': 0.5,
                    'tag': 'parser_false'
                }
            
            # 查找对应的概率项(根据idx或id匹配)
            prob_item = None
            
            # 如果传入了idx，先尝试索引匹配
            if 0 <= idx < len(batch_probs):
                candidate = batch_probs[idx]
                if isinstance(candidate, dict) and 'id' in candidate:
                    # 验证id是否匹配(考虑字符串数字转换)
                    try:
                        if str(candidate['id']) == str(idx):
                            prob_item = candidate
                    except:
                        pass
                
                # 如果索引匹配失败，直接使用索引位置的项
                if prob_item is None:
                    prob_item = candidate
            
            # 如果还没找到，遍历查找第一个匹配的id
            if prob_item is None:
                for item in batch_probs:
                    if isinstance(item, dict) and 'id' in item:
                        try:
                            if str(item['id']) == str(idx):
                                prob_item = item
                                break
                        except:
                            continue
            
            # 如果还没找到，使用第一个可用项
            if prob_item is None and len(batch_probs) > 0:
                prob_item = batch_probs[0]
            
            if prob_item is None or not isinstance(prob_item, dict):
                return {
                    'id': sample_id,
                    'pred_label': default_label,
                    'auc_prob': 0.5,
                    'tag': 'parser_false'
                }
            
            # 解析label_probs
            label_probs = prob_item.get('label_probs', [])
            if not isinstance(label_probs, list):
                return {
                    'id': sample_id,
                    'pred_label': default_label,
                    'auc_prob': 0.5,
                    'tag': 'parser_false'
                }
            
            # 找到概率最高的有效数字标签
            max_prob = -1
            best_label = default_label
            prob_dict = {}
            
            for lp in label_probs:
                if isinstance(lp, dict) and 'label' in lp and 'prob' in lp:
                    try:
                        label = str(int(lp['label']))  # 验证是数字标签
                        prob = float(lp['prob'])
                        prob_dict[label] = prob
                        
                        if prob > max_prob:
                            max_prob = prob
                            best_label = label
                    except (ValueError, TypeError):
                        continue
            
            # 计算正类概率
            if '1' in prob_dict:
                auc_prob = prob_dict['1']
            elif '0' in prob_dict:
                auc_prob = 1.0 - prob_dict['0']
            else:
                auc_prob = 0.5
            
            # 如果找到了有效标签
            if max_prob > 0:
                return {
                    'id': sample_id,
                    'pred_label': best_label,
                    'auc_prob': auc_prob,
                    'tag': 'parser_success'
                }
            else:
                return {
                    'id': sample_id,
                    'pred_label': default_label,
                    'auc_prob': 0.5,
                    'tag': 'parser_false'
                }
                
        except Exception as e:
            logger.debug(f"独立函数b解析失败: {e}")
            return {
                'id': sample_id,
                'pred_label': default_label,
                'auc_prob': 0.5,
                'tag': 'parser_false'
            }
    
    # 以下是原有的函数，保留兼容性
    
    @staticmethod
    def _extract_label_from_probabilities(data: Dict[str, Any], sample_idx: int, default_label: str) -> Tuple[str, bool]:
        """从batch_probabilities字段提取标签
        
        Args:
            data: 包含batch_probabilities的数据
            sample_idx: 样本索引
            default_label: 默认标签
            
        Returns:
            Tuple[str, bool]: (预测标签, 是否成功提取)
        """
        try:
            # 检查是否有batch_probabilities字段
            if 'batch_probabilities' not in data:
                return default_label, False
            
            batch_probs = data['batch_probabilities']
            if not isinstance(batch_probs, list) or sample_idx >= len(batch_probs):
                return default_label, False
            
            sample_prob = batch_probs[sample_idx]
            if not isinstance(sample_prob, dict) or 'label_probs' not in sample_prob:
                return default_label, False
            
            label_probs = sample_prob['label_probs']
            if not isinstance(label_probs, list):
                return default_label, False
            
            # 找到概率最高的标签
            max_prob = -1
            best_label = default_label
            
            for prob_item in label_probs:
                if isinstance(prob_item, dict) and 'label' in prob_item and 'prob' in prob_item:
                    try:
                        label = str(prob_item['label'])
                        prob = float(prob_item['prob'])
                        
                        # 验证是否为有效的数值标签
                        int(label)  # 验证标签是数值
                        
                        if prob > max_prob:
                            max_prob = prob
                            best_label = label
                    except (ValueError, TypeError):
                        # 标签不是数值，跳过
                        continue
            
            # 如果找到了有效的标签并且概率大于0
            if max_prob > 0:
                return best_label, True
            else:
                return default_label, False
                
        except Exception as e:
            logger.debug(f"概率分析失败: {e}")
            return default_label, False

    @staticmethod
    def _extract_label_and_prob_from_probabilities(data: Dict[str, Any], sample_idx: int, default_label: str) -> Tuple[str, float, bool]:
        """从batch_probabilities字段提取标签和概率
        
        Args:
            data: 包含batch_probabilities的数据
            sample_idx: 样本索引
            default_label: 默认标签
            
        Returns:
            Tuple[str, float, bool]: (预测标签, 正类概率, 是否成功提取)
        """
        try:
            # 检查是否有batch_probabilities字段
            if 'batch_probabilities' not in data:
                return default_label, 0.5, False
            
            batch_probs = data['batch_probabilities']
            if not isinstance(batch_probs, list) or sample_idx >= len(batch_probs):
                return default_label, 0.5, False
            
            sample_prob = batch_probs[sample_idx]
            if not isinstance(sample_prob, dict) or 'label_probs' not in sample_prob:
                return default_label, 0.5, False
            
            label_probs = sample_prob['label_probs']
            if not isinstance(label_probs, list):
                return default_label, 0.5, False
            
            # 找到概率最高的标签，同时记录所有概率
            max_prob = -1
            best_label = default_label
            prob_dict = {}
            
            for prob_item in label_probs:
                if isinstance(prob_item, dict) and 'label' in prob_item and 'prob' in prob_item:
                    try:
                        label = str(prob_item['label'])
                        prob = float(prob_item['prob'])
                        
                        # 验证是否为有效的数值标签
                        int(label)  # 验证标签是数值
                        
                        prob_dict[label] = prob
                        if prob > max_prob:
                            max_prob = prob
                            best_label = label
                    except (ValueError, TypeError):
                        # 标签不是数值，跳过
                        continue
            
            # 对于二分类，返回正类(标签为1)的概率
            positive_class_prob = prob_dict.get('1', 0.5)
            if len(prob_dict) == 2 and '0' in prob_dict and '1' in prob_dict:
                # 标准二分类情况，返回正类概率
                auc_prob = positive_class_prob
            else:
                # 其他情况，如果预测是正类则用其概率，否则用1-负类概率
                if best_label == '1':
                    auc_prob = max_prob
                elif best_label == '0' and '0' in prob_dict:
                    auc_prob = 1.0 - prob_dict['0']
                else:
                    auc_prob = 0.5
            
            # 如果找到了有效的标签并且概率大于0，就认为推断成功
            if max_prob > 0:
                return best_label, auc_prob, True
            else:
                return default_label, 0.5, False
                
        except Exception as e:
            logger.debug(f"概率分析失败: {e}")
            return default_label, 0.5, False

    @staticmethod
    def _smart_recover_from_parsed_data(data: Dict[str, Any], line_num: int, default_label: str) -> List[Tuple[str, str, str, float]]:
        """
        从已解析的JSON数据中智能恢复批量数据（优化版，无需正则）
        
        策略：
        1. 直接从已解析的data['groundtruth']获取真实标签列表
        2. 直接从已解析的data['batch_probabilities']获取概率信息
        3. 对每个位置进行概率推断，成功标记为'real'，失败标记为'default'
        
        Args:
            data: 已解析的JSON数据，包含groundtruth和batch_probabilities字段
            line_num: 行号（用于日志）
            default_label: 默认标签
            
        Returns:
            List[Tuple[str, str, str, float]]: [(true_label, pred_label, tag, auc_prob), ...]
        """
        recovered_samples = []
        
        try:
            # 1. 直接获取groundtruth数据
            if 'groundtruth' not in data:
                logger.warning(f"第{line_num}行缺少groundtruth字段，使用单个默认样本")
                return [(default_label, default_label, 'default', 0.5)]
            
            # 解析groundtruth（可能是字符串也可能已经是对象）
            groundtruth_data = data['groundtruth']
            if isinstance(groundtruth_data, str):
                groundtruth_data = json.loads(groundtruth_data)
            
            if not isinstance(groundtruth_data, list):
                logger.warning(f"第{line_num}行groundtruth不是列表格式")
                return [(default_label, default_label, 'default', 0.5)]
            
            # 2. 直接获取batch_probabilities数据
            batch_probs_data = data.get('batch_probabilities', [])
            if isinstance(batch_probs_data, str):
                batch_probs_data = json.loads(batch_probs_data)
            
            if not isinstance(batch_probs_data, list):
                logger.warning(f"第{line_num}行batch_probabilities不是列表格式")
                batch_probs_data = []
            
            logger.debug(f"第{line_num}行成功获取到{len(batch_probs_data)}个概率项")
            
            # 3. 按顺序处理每个样本
            used_prob_indices = set()  # 记录已使用的概率项索引，避免重复使用同一个概率项
            real_inference_items = []  # 记录真实推断的项目
            default_filling_items = []  # 记录默认填充的项目
            
            for idx, truth_item in enumerate(groundtruth_data):
                if not isinstance(truth_item, dict) or 'label' not in truth_item:
                    logger.warning(f"第{line_num}行第{idx}项真实标签格式异常")
                    sample_id = str(idx)
                    recovered_samples.append((default_label, default_label, 'default', 0.5))
                    default_filling_items.append((line_num, sample_id))
                    continue
                
                true_label = str(truth_item['label'])
                sample_id = str(truth_item.get('id', idx))  # 使用id字段，如果没有则用索引
                
                # 4. 尝试从batch_probabilities中找到对应的概率信息
                pred_label = default_label
                tag = 'default'
                auc_prob = 0.5  # 默认概率
                
                # 查找第一个匹配且未使用的概率项
                found_match = False
                for prob_idx, prob_item in enumerate(batch_probs_data):
                    if (prob_idx not in used_prob_indices and
                        isinstance(prob_item, dict) and 
                        'id' in prob_item and 
                        str(prob_item['id']) == sample_id):
                        
                        logger.debug(f"第{line_num}行样本{sample_id}找到匹配的概率项，索引{prob_idx}")
                        
                        # 尝试概率推断（使用新函数同时获取标签和概率）
                        inferred_label, inferred_prob, inference_success = TaggedSingleFileEvaluationLayer._extract_label_and_prob_from_probabilities(
                            {'batch_probabilities': [prob_item]}, 0, default_label
                        )
                        
                        used_prob_indices.add(prob_idx)  # 标记这个概率项已使用
                        found_match = True
                        
                        if inference_success:
                            pred_label = inferred_label
                            auc_prob = inferred_prob
                            tag = 'real'
                            real_inference_items.append((line_num, sample_id))
                            logger.debug(f"第{line_num}行样本{sample_id}概率推断成功: {pred_label}")
                        else:
                            logger.debug(f"第{line_num}行样本{sample_id}概率推断失败，使用默认标签")
                        break
                
                if not found_match:
                    logger.debug(f"第{line_num}行样本{sample_id}未找到匹配的概率项")
                
                # 记录结果
                recovered_samples.append((true_label, pred_label, tag, auc_prob))
                if tag == 'default':
                    default_filling_items.append((line_num, sample_id))
            
            # 详细日志输出
            real_count = sum(1 for _, _, tag in recovered_samples if tag == 'real')
            default_count = sum(1 for _, _, tag in recovered_samples if tag == 'default')
            
            logger.info(f"第{line_num}行智能恢复: 总样本{len(recovered_samples)}，真实预测{real_count}，默认填充{default_count}")
            
            if real_inference_items:
                real_info = ", ".join([f"行{line}:id{sid}" for line, sid in real_inference_items[:10]])
                logger.info(f"  真实推断成功: {real_info}{'...' if len(real_inference_items) > 10 else ''}")
            
            if default_filling_items:
                default_info = ", ".join([f"行{line}:id{sid}" for line, sid in default_filling_items[:10]])
                logger.warning(f"  默认填充项目: {default_info}{'...' if len(default_filling_items) > 10 else ''}")
            
            return recovered_samples
            
        except Exception as e:
            logger.warning(f"第{line_num}行智能恢复失败: {e}，使用单个默认样本")
            return [(default_label, default_label, 'default', 0.5)]

    @staticmethod
    def _smart_recover_batch_data_from_failed_line(line: str, line_num: int, default_label: str) -> List[Tuple[str, str, str]]:
        """
        从解析失败的行中智能恢复批量数据
        
        策略：
        1. 从groundtruth字段获取真实标签列表和数量
        2. 从batch_probabilities字段按顺序提取概率信息
        3. 对每个位置进行概率推断，成功标记为'real'，失败标记为'default'
        4. 避免重复标签，每个id只处理第一次出现的
        
        Args:
            line: 解析失败的原始行数据
            line_num: 行号（用于日志）
            default_label: 默认标签
            
        Returns:
            List[Tuple[str, str, str]]: [(true_label, pred_label, tag), ...]
        """
        
        recovered_samples = []
        
        try:
            # 1. 提取groundtruth字段
            groundtruth_pattern = r'"groundtruth":\s*"(\[.*?\])"'
            groundtruth_match = re.search(groundtruth_pattern, line)
            
            if not groundtruth_match:
                logger.warning(f"第{line_num}行无法找到groundtruth字段，使用单个默认样本")
                return [(default_label, default_label, 'default', 0.5)]
            
            # 解析groundtruth
            groundtruth_str = groundtruth_match.group(1).replace('\\"', '"')
            groundtruth_data = json.loads(groundtruth_str)
            
            if not isinstance(groundtruth_data, list):
                logger.warning(f"第{line_num}行groundtruth不是列表格式")
                return [(default_label, default_label, 'default', 0.5)]
            
            # 2. 提取batch_probabilities字段
            batch_probs_pattern = r'"batch_probabilities":\s*(\[.*?\])(?=,\s*"[^"]+"|$)'
            batch_probs_match = re.search(batch_probs_pattern, line, re.DOTALL)
            
            batch_probs_data = []
            if batch_probs_match:
                try:
                    batch_probs_str = batch_probs_match.group(1)
                    batch_probs_data = json.loads(batch_probs_str)
                    logger.debug(f"第{line_num}行成功提取到{len(batch_probs_data)}个概率项")
                except json.JSONDecodeError as e:
                    logger.warning(f"第{line_num}行batch_probabilities解析失败: {e}")
                    batch_probs_data = []
            else:
                logger.warning(f"第{line_num}行无法找到batch_probabilities字段")
            
            # 3. 按顺序处理每个样本
            used_prob_indices = set()  # 记录已使用的概率项索引，避免重复使用同一个概率项
            real_inference_items = []  # 记录真实推断的项目
            default_filling_items = []  # 记录默认填充的项目
            
            for idx, truth_item in enumerate(groundtruth_data):
                if not isinstance(truth_item, dict) or 'label' not in truth_item:
                    logger.warning(f"第{line_num}行第{idx}项真实标签格式异常")
                    sample_id = str(idx)
                    recovered_samples.append((default_label, default_label, 'default', 0.5))
                    default_filling_items.append((line_num, sample_id))
                    continue
                
                true_label = str(truth_item['label'])
                sample_id = str(truth_item.get('id', idx))  # 使用id字段，如果没有则用索引
                
                # 4. 尝试从batch_probabilities中找到对应的概率信息
                # 按顺序查找第一个未使用的匹配项
                pred_label = default_label
                tag = 'default'
                
                # 查找第一个匹配且未使用的概率项
                found_match = False
                for prob_idx, prob_item in enumerate(batch_probs_data):
                    if (prob_idx not in used_prob_indices and
                        isinstance(prob_item, dict) and 
                        'id' in prob_item and 
                        str(prob_item['id']) == sample_id):
                        
                        logger.debug(f"第{line_num}行样本{sample_id}找到匹配的概率项，索引{prob_idx}")
                        
                        # 尝试概率推断
                        inferred_label, inference_success = TaggedSingleFileEvaluationLayer._extract_label_from_probability_item(
                            prob_item, default_label
                        )
                        
                        used_prob_indices.add(prob_idx)  # 标记这个概率项已使用
                        found_match = True
                        
                        if inference_success:
                            pred_label = inferred_label
                            tag = 'real'
                            real_inference_items.append((line_num, sample_id))
                            logger.debug(f"第{line_num}行样本{sample_id}概率推断成功: {pred_label}")
                        else:
                            logger.debug(f"第{line_num}行样本{sample_id}概率推断失败，使用默认标签")
                        break
                
                if not found_match:
                    logger.debug(f"第{line_num}行样本{sample_id}未找到匹配的概率项")
                
                # 记录结果
                recovered_samples.append((true_label, pred_label, tag))
                if tag == 'default':
                    default_filling_items.append((line_num, sample_id))
            
            # 详细日志输出
            real_count = sum(1 for _, _, tag in recovered_samples if tag == 'real')
            default_count = sum(1 for _, _, tag in recovered_samples if tag == 'default')
            
            logger.info(f"第{line_num}行智能恢复: 总样本{len(recovered_samples)}，真实预测{real_count}，默认填充{default_count}")
            
            if real_inference_items:
                real_info = ", ".join([f"行{line}:id{sid}" for line, sid in real_inference_items[:10]])
                logger.info(f"  真实推断成功: {real_info}{'...' if len(real_inference_items) > 10 else ''}")
            
            if default_filling_items:
                default_info = ", ".join([f"行{line}:id{sid}" for line, sid in default_filling_items[:10]])
                logger.warning(f"  默认填充项目: {default_info}{'...' if len(default_filling_items) > 10 else ''}")
            
            return recovered_samples
            
        except Exception as e:
            logger.warning(f"第{line_num}行智能恢复失败: {e}，使用单个默认样本")
            return [(default_label, default_label, 'default', 0.5)]

    @staticmethod
    def _extract_label_from_probability_item(prob_item: dict, default_label: str) -> Tuple[str, bool]:
        """
        从单个概率项中提取标签
        
        Args:
            prob_item: 概率项，格式如 {"id": "0", "label_probs": [{"label": "0", "prob": 0.95}, ...]}
            default_label: 默认标签
            
        Returns:
            Tuple[str, bool]: (预测标签, 是否成功提取)
        """
        try:
            if 'label_probs' not in prob_item:
                return default_label, False
            
            label_probs = prob_item['label_probs']
            if not isinstance(label_probs, list):
                return default_label, False
            
            # 找到概率最高的标签
            max_prob = -1
            best_label = default_label
            
            for prob_entry in label_probs:
                if isinstance(prob_entry, dict) and 'label' in prob_entry and 'prob' in prob_entry:
                    try:
                        label = str(prob_entry['label'])
                        prob = float(prob_entry['prob'])
                        
                        # 验证是否为有效的数值标签
                        int(label)  # 验证标签是数值
                        
                        if prob > max_prob:
                            max_prob = prob
                            best_label = label
                    except (ValueError, TypeError):
                        # 标签不是数值，跳过
                        continue
            
            # 如果找到了有效的标签并且概率大于0，就认为推断成功
            # 即使推断结果等于默认标签，也是基于概率的成功推断
            if max_prob > 0:
                return best_label, True
            else:
                return default_label, False
                
        except Exception:
            return default_label, False
