#!/usr/bin/env python3
"""
Layer 4: 智能投票聚合层

职责：
1. 接收多个文件的评估结果（带标记）
2. 如果只有1个文件：直接透传结果
3. 如果有多个文件：进行智能投票聚合
4. 投票时只使用真实预测，忽略默认填充值
5. 返回统一格式的最终结果

输入：Dict[str, dict] - {文件名: 评估结果}
输出：dict - 最终聚合结果
"""

import os
import logging
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

from sklearn.metrics import classification_report, roc_auc_score



class SmartVotingAggregationLayer:
    """智能投票聚合层 - 负责单文件透传或多文件智能投票聚合"""
    
    @staticmethod
    def aggregate_results(file_results: Dict[str, dict], weighted: bool = True) -> dict:
        """
        智能聚合文件评估结果
        
        Args:
            file_results: {文件名: 评估结果}
            weighted: 是否使用加权投票
            
        Returns:
            dict: 最终聚合结果
        """
        # 过滤出有效的结果（有数据的结果）
        successful_results = {path: result for path, result in file_results.items() 
                            if result.get('total_samples', 0) > 0}
        
        if not successful_results:
            raise Exception("没有有效的评估结果可以聚合")
        
        if len(successful_results) == 1:
            # 单文件模式：直接透传
            return SmartVotingAggregationLayer._handle_single_file(successful_results)
        else:
            # 多文件模式：智能投票聚合
            return SmartVotingAggregationLayer._handle_multiple_files(successful_results, weighted)
    
    @staticmethod
    def _handle_single_file(successful_results: Dict[str, dict]) -> dict:
        """
        处理单文件结果（直接透传）
        
        Args:
            successful_results: 包含单个成功结果的字典
            
        Returns:
            dict: 透传的结果，添加聚合标识
        """
        file_path, result = next(iter(successful_results.items()))
        
        # 添加聚合信息
        final_result = result.copy()
        final_result.update({
            'aggregation_mode': 'single_file',
            'file_count': 1,
            'source_files': [file_path],
            'dataset_name': SmartVotingAggregationLayer._extract_dataset_name(file_path)
        })
        
        print(f"📄 单文件模式: {os.path.basename(file_path)}")
        return final_result
    
    @staticmethod
    def _handle_multiple_files(successful_results: Dict[str, dict], weighted: bool) -> dict:
        """
        处理多文件结果（智能投票聚合）
        
        Args:
            successful_results: 多个成功结果的字典
            weighted: 是否使用加权投票
            
        Returns:
            dict: 智能投票聚合后的结果
        """
        print(f"VOTE:  多文件智能投票模式: {len(successful_results)} 个文件")
        
        # 收集所有文件的数据和标记
        all_predictions_with_tags = []
        all_probabilities_with_tags = []
        all_ground_truth = None
        voting_stats = {
            'total_positions': 0,
            'positions_with_all_real': 0,
            'positions_with_partial_real': 0,
            'positions_with_no_real': 0,
            'total_real_votes': 0,
            'total_default_votes': 0
        }
        
        for file_path, result in successful_results.items():
            # 获取当前文件的数据
            predictions = result.get('y_pred', [])
            ground_truth = result.get('y_true', [])
            prediction_tags = result.get('tags', [])
            file_probs = result.get('y_probs', [])  # 尝试获取概率信息
            
            # 如果没有概率信息，使用0.5填充
            if not file_probs or len(file_probs) != len(predictions):
                logger.warning(f"文件 {result.get('file_path', '')} 缺少概率信息，使用0.5填充")
                auc_probs = [0.5] * len(predictions)
            else:
                auc_probs = file_probs
            
            if all_ground_truth is None:
                all_ground_truth = ground_truth
            
            # 将预测、概率和标记组合在一起
            predictions_with_tags = list(zip(predictions, prediction_tags))
            probabilities_with_tags = list(zip(auc_probs, prediction_tags))
            
            all_predictions_with_tags.append(predictions_with_tags)
            all_probabilities_with_tags.append(probabilities_with_tags)
        
        if not all_predictions_with_tags or all_ground_truth is None:
            raise Exception("智能投票聚合缺少必要的预测数据")
        
        # 执行智能投票
        final_predictions, final_auc_probs = SmartVotingAggregationLayer._smart_voting(
            all_predictions_with_tags, all_probabilities_with_tags, weighted, voting_stats
        )
        
        # 重要：确保ground truth长度与最终预测长度一致
        # 如果投票过程中截断了长度，也需要截断ground truth
        truncated_ground_truth = all_ground_truth[:len(final_predictions)] if all_ground_truth else []
        
        # 生成最终报告
        report, auc_score = SmartVotingAggregationLayer._generate_aggregated_report(
            final_predictions, truncated_ground_truth, final_auc_probs
        )
        
        # 构建聚合结果
        final_result = {
            'aggregation_mode': 'smart_voting',
            'voting_method': 'weighted' if weighted else 'majority',
            'file_count': len(successful_results),
            'source_files': list(successful_results.keys()),
            'classification_report': report,
            'auc_score': auc_score,
            'y_pred': final_predictions,
            'y_true': truncated_ground_truth,  # 使用截断后的ground truth
            'total_samples': len(final_predictions),
            'real_predictions': voting_stats['total_real_votes'],
            'default_fillings': voting_stats['total_default_votes'],
            'voting_statistics': voting_stats
        }
        
        # 打印投票统计信息
        print(f"INFO: 投票统计:")
        print(f"   总投票位置: {voting_stats['total_positions']}")
        print(f"   全真实投票位置: {voting_stats['positions_with_all_real']}")
        print(f"   部分真实投票位置: {voting_stats['positions_with_partial_real']}")
        print(f"   无真实投票位置: {voting_stats['positions_with_no_real']}")
        print(f"   真实投票总数: {voting_stats['total_real_votes']}")
        print(f"   默认投票总数: {voting_stats['total_default_votes']}")
        
        return final_result
    
    @staticmethod
    def _smart_voting(predictions_with_tags: List[List[Tuple]], probabilities_with_tags: List[List[Tuple]], 
                     weighted: bool, voting_stats: dict) -> Tuple[List, List]:
        """
        智能投票：只使用真实预测参与投票，实现鲁棒性处理
        
        处理逻辑：
        1. 智能处理不同长度的预测列表（因为解析错误导致的长度不一致）
        2. 对每个位置，只使用tag="parser_success"的预测参与投票
        3. 如果某位置全部都是tag="default"，则直接使用默认值
        4. 支持加权投票和简单多数投票两种模式
        
        Args:
            predictions_with_tags: [(预测, 标记), ...] 的列表，每个文件的预测和标记
            probabilities_with_tags: [(概率, 标记), ...] 的列表，每个文件的概率和标记
            weighted: 是否使用加权投票
            voting_stats: 投票统计信息
            
        Returns:
            Tuple[List, List]: (最终预测, 最终概率)
        """
        if not predictions_with_tags:
            return [], []
        
        # 智能处理不同长度的预测列表：使用最短长度作为基准
        # 这是因为某些文件可能由于JSON解析错误而跳过了一些行
        lengths = [len(preds) for preds in predictions_with_tags]
        length = min(lengths)
        
        # 如果长度不一致，打印警告并截断到最短长度
        if len(set(lengths)) > 1:
            print(f"WARNING:  警告: 发现不同长度的预测列表 {lengths}，将使用最短长度 {length} 进行投票")
            print(f"    这通常是由于某些文件存在JSON解析错误导致的")
            # 截断所有列表到最短长度，确保投票时每个位置都有对应的预测
            predictions_with_tags = [preds[:length] for preds in predictions_with_tags]
            probabilities_with_tags = [probs[:length] for probs in probabilities_with_tags]
        
        final_predictions = []
        final_auc_probs = []
        
        voting_stats['total_positions'] = length
        
        # 对每个位置进行投票
        for i in range(length):
            # 收集第i个位置的所有真实预测（tag="parser_success"）
            real_votes = []
            real_probs = []
            default_votes = []
            
            for voter_idx in range(len(predictions_with_tags)):
                pred, tag = predictions_with_tags[voter_idx][i]
                prob, prob_tag = probabilities_with_tags[voter_idx][i]
                
                if tag == "parser_success":
                    # 只有真实预测才参与投票
                    real_votes.append(pred)
                    real_probs.append(prob)
                    voting_stats['total_real_votes'] += 1
                else:
                    # 收集默认值，备用
                    default_votes.append(pred)
                    voting_stats['total_default_votes'] += 1
            
            # 根据真实投票数量进行决策
            if len(real_votes) == 0:
                # 情况4：所有投票都是默认值（m-n=0），直接使用默认值
                if default_votes:
                    # 使用最常见的默认值
                    vote_counts = Counter(default_votes)
                    most_common_default = vote_counts.most_common(1)[0][0]
                    final_predictions.append(int(most_common_default))
                else:
                    # 极端情况的兜底
                    final_predictions.append(0)
                final_auc_probs.append(0.5)  # 默认概率
                voting_stats['positions_with_no_real'] += 1
                
            else:
                # 有真实投票，进行正常投票（情况4：使用m-n个真实预测投票）
                if weighted and len(real_probs) > 0:
                    # 加权投票：使用概率信息
                    final_pred = SmartVotingAggregationLayer._weighted_vote_single_position(
                        real_votes, real_probs
                    )
                else:
                    # 简单多数投票：只看票数
                    vote_counts = Counter(real_votes)
                    final_pred = vote_counts.most_common(1)[0][0]
                
                final_predictions.append(int(final_pred))
                
                # 计算平均概率
                avg_prob = sum(real_probs) / len(real_probs) if real_probs else 0.5
                final_auc_probs.append(avg_prob)
                
                # 统计投票类型
                if len(real_votes) == len(predictions_with_tags):
                    voting_stats['positions_with_all_real'] += 1
                else:
                    voting_stats['positions_with_partial_real'] += 1
        
        return final_predictions, final_auc_probs
    
    @staticmethod
    def _weighted_vote_single_position(votes: List, probs: List) -> int:
        """
        对单个位置进行加权投票
        
        Args:
            votes: 投票列表
            probs: 概率列表
            
        Returns:
            int: 投票结果
        """
        if not votes:
            return 0
        
        # 简化的加权投票：使用概率作为权重
        weighted_votes = defaultdict(float)
        
        for vote, prob in zip(votes, probs):
            # 使用概率的置信度作为权重
            confidence = abs(prob - 0.5) * 2  # 将[0,1]映射到[1,0,1]的置信度
            weighted_votes[vote] += confidence
        
        # 返回权重最高的投票
        if weighted_votes:
            return max(weighted_votes, key=weighted_votes.get)
        else:
            return Counter(votes).most_common(1)[0][0]
    
    @staticmethod
    def _generate_aggregated_report(predictions: list, ground_truth: list, auc_probabilities: list) -> Tuple[str, float]:
        """
        生成聚合后的分类报告
        
        Args:
            predictions: 最终预测结果
            ground_truth: 真实标签
            auc_probabilities: AUC计算用概率
            
        Returns:
            Tuple[str, float]: (分类报告, AUC分数)
        """
        try:
            # 确保所有标签都是整数类型，避免类型混合错误
            predictions_int = [int(pred) for pred in predictions]
            ground_truth_int = [int(label) for label in ground_truth]
            
            unique_labels = sorted(set(ground_truth_int))
            
            # 生成分类报告
            report = classification_report(
                ground_truth_int, predictions_int, 
                labels=unique_labels, 
                target_names=[f"Class {label}" for label in unique_labels],
                digits=4, zero_division=0
            )
            
            # 计算AUC（仅适用于二分类）
            auc_score = None
            if len(unique_labels) == 2:
                try:
                    auc_score = roc_auc_score(ground_truth_int, auc_probabilities)
                except Exception as e:
                    print(f"WARNING: 聚合AUC计算失败: {e}")
                    
        except Exception as e:
            print(f"ERROR: 聚合报告生成失败: {e}")
            report = f"聚合评估失败: {e}"
            auc_score = None
            
        return report, auc_score
    
    @staticmethod
    def _extract_dataset_name(file_path: str) -> str:
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
            parts = filename.split('@@')
            if len(parts) >= 2:
                second_part = parts[1]
                if '_' in second_part:
                    return second_part.split('_')[0]
        
        # 备选方案
        if '_' in filename:
            return filename.split('_')[0]
        
        return filename.replace('.jsonl', '')
