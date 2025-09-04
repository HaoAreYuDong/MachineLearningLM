#!/usr/bin/env python3
"""
Layer 5: 输出层
"""

import os
import json
from typing import Dict, Any


class OutputLayer:
    """输出层 - 负责格式化和输出最终结果"""
    
    @staticmethod
    def format_and_output(result: Dict[str, Any], output_dir: str, dataset_name: str, 
                         model_name: str, row_shuffle_seeds: str) -> Dict[str, str]:
        """
        格式化并输出最终结果
        
        Args:
            result: 聚合后的评估结果
            output_dir: 输出目录
            dataset_name: 数据集名称
            model_name: 模型名称
            row_shuffle_seeds: 随机种子
            
        Returns:
            Dict[str, str]: 输出文件路径信息
        """
        # 处理模型名称：只取路径的最后一部分，并处理特殊字符
        model_name_clean = model_name or 'auto_detected'
        
        # 处理 backend::model 格式
        if '::' in model_name_clean:
            parts = model_name_clean.split('::', 1)
            backend, actual_model = parts[0], parts[1]
            if backend.lower() == 'openai':
                # 对于 openai::model，使用后面的模型名
                model_name_clean = actual_model
            else:
                # 对于其他 backend，使用完整的格式
                model_name_clean = model_name_clean.replace('::', '_')
        
        # 处理 HuggingFace 格式的路径（如 minzl/toy_3550）
        if '/' in model_name_clean:
            model_name_clean = model_name_clean.split('/')[-1]
        
        # 替换所有可能的特殊字符为 @ 符号
        safe_model_name = model_name_clean.replace('-', '_').replace('.', '_').replace(':', '@').replace('/', '_')
        
        # 处理 row_shuffle_seeds，移除方括号和空格，用 @ 替换特殊字符
        safe_seeds = str(row_shuffle_seeds).replace('[', '@').replace(']', '@').replace(' ', '').replace(',', '_').replace(':', '@')
        
        # 构建完整输出目录路径 - 修改为 dataset_name/model_name 的层级结构
        model_output_dir = os.path.join(output_dir, dataset_name, safe_model_name)
        
        # 确保输出目录存在
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 从结果中提取 train_chunk_size 和 test_chunk_size（如果有的话）
        train_size = result.get('train_chunk_size', 'unknown')
        test_size = result.get('test_chunk_size', 'unknown')
        split_seed = result.get('split_seed', 'unknown')
        
        # 生成输出文件名 - 新格式: model_name@@dataset_Sseed*_trainsize*_testsize*_seed@*_report.json/txt
        base_filename = f"{safe_model_name}@@{dataset_name}_Sseed{split_seed}_trainsize{train_size}_testsize{test_size}_seed{safe_seeds}_report"
        json_output_path = os.path.join(model_output_dir, f"{base_filename}.json")
        txt_output_path = os.path.join(model_output_dir, f"{base_filename}.txt")
        
        # 格式化结果
        formatted_result = OutputLayer._format_result(result, dataset_name, model_name, row_shuffle_seeds)
        
        # 写入 JSON 文件
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_result, f, indent=2, ensure_ascii=False)
        
        # 写入 TXT 文件
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write(OutputLayer._format_txt_report(formatted_result))
        
        # 打印总结
        OutputLayer._print_summary(formatted_result)
        
        print(f"\nSAVE: 评估结果已保存到:")
        print(f"   📄 JSON: {json_output_path}")
        print(f"   📄 TXT:  {txt_output_path}")
        
        return {
            'json_file': json_output_path,
            'txt_file': txt_output_path,
            'dataset_name': dataset_name,
            'model_name': model_name,
            'seeds': row_shuffle_seeds
        }
    
    @staticmethod
    def _format_result(result: Dict[str, Any], dataset_name: str, model_name: str, seeds: str) -> Dict[str, Any]:
        """格式化结果 - 参考 b41.json 格式"""
        # 获取源文件信息
        source_files = result.get('source_files', [])
        input_file = source_files[0] if source_files else "Unknown"
        
        # 处理分类报告
        classification_report_str = result.get('classification_report', 'N/A')
        class_metrics = {}
        macro_avg_precision = None
        macro_avg_recall = None
        macro_avg_f1 = None
        macro_avg_support = None
        weighted_avg_precision = None
        weighted_avg_recall = None
        weighted_avg_f1 = None
        weighted_avg_support = None
        
        # 解析分类报告字符串，提取各类别指标
        if classification_report_str and classification_report_str != 'N/A':
            try:
                lines = classification_report_str.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('Class '):
                        # 解析类别行：如 "Class 0     0.8903    0.9512    0.9198      7978"
                        parts = line.split()
                        if len(parts) >= 5:
                            class_name = f"class_{parts[1]}"
                            precision = float(parts[2])
                            recall = float(parts[3])
                            f1_score = float(parts[4])
                            support = int(parts[5])
                            
                            class_metrics[class_name] = {
                                "precision": precision,
                                "recall": recall,
                                "f1-score": f1_score,
                                "support": support
                            }
                    elif line.startswith('macro avg'):
                        # 解析 macro avg 行
                        parts = line.split()
                        if len(parts) >= 5:
                            macro_avg_precision = float(parts[2])
                            macro_avg_recall = float(parts[3])
                            macro_avg_f1 = float(parts[4])
                            macro_avg_support = int(parts[5])
                    elif line.startswith('weighted avg'):
                        # 解析 weighted avg 行
                        parts = line.split()
                        if len(parts) >= 5:
                            weighted_avg_precision = float(parts[2])
                            weighted_avg_recall = float(parts[3])
                            weighted_avg_f1 = float(parts[4])
                            weighted_avg_support = int(parts[5])
            except Exception as e:
                print(f"WARNING: 解析分类报告时出错: {e}")
        
        # 处理 AUC Score
        auc_score = result.get('auc_score', None)
        auc_message = auc_score if auc_score is not None else "Not calculated (requires binary classification)"
        
        # 生成类似 b41.json 的格式
        formatted = {
            "Dataset": dataset_name,
            "Input file": input_file,
            "Total samples": result.get('total_samples', 0),
            "Responses with wrong sample size": 0,  # 假设默认为0，可以后续添加此统计
            "Evaluation mode": result.get('aggregation_mode', 'unknown').replace('_', ' ').title(),
            "AUC Score": auc_message,
            "class_metrics": class_metrics
        }
        
        # 添加 macro 和 weighted 平均值（如果解析成功）
        if macro_avg_precision is not None:
            formatted["macro_avg_precision"] = macro_avg_precision
            formatted["macro_avg_recall"] = macro_avg_recall
            formatted["macro_avg_f1"] = macro_avg_f1
            formatted["macro_avg_support"] = macro_avg_support
            
        if weighted_avg_precision is not None:
            formatted["weighted_avg_precision"] = weighted_avg_precision
            formatted["weighted_avg_recall"] = weighted_avg_recall
            formatted["weighted_avg_f1"] = weighted_avg_f1
            formatted["weighted_avg_support"] = weighted_avg_support
        
        return formatted
    
    @staticmethod
    def _print_summary(result: Dict[str, Any]):
        """打印评估总结 - 与TXT文件格式保持一致"""
        print("\n" + "-" * 70)
        print("EVALUATION SUMMARY")
        print("-" * 70)
        
        # 基本信息 - 简洁格式，与TXT一致
        print(f"Dataset: {result.get('Dataset', 'N/A')}")
        
        # 从输入文件路径中提取模型名称
        input_file = result.get('Input file', '')
        if '@@' in input_file:
            model_part = input_file.split('@@')[0].split('/')[-1] if input_file else 'N/A'
        else:
            model_part = 'N/A'
        print(f"Model: {model_part}")
        
        print(f"Evaluation Mode: {result.get('Evaluation mode', 'N/A')}")
        print(f"Total samples: {result.get('Total samples', 0)}")
        print(f"Responses with wrong sample size: {result.get('Responses with wrong sample size', 0)}")
        
        # AUC Score
        auc_score = result.get('AUC Score')
        if auc_score is not None and isinstance(auc_score, (int, float)):
            print(f"AUC Score: {auc_score:.4f}")
        else:
            print(f"AUC Score: {auc_score}")
        
        print("-" * 70)
        print()
        
        # 重新构建分类报告的原始格式 - 与TXT完全一致
        class_metrics = result.get('class_metrics', {})
        if class_metrics:
            # 表头
            print("              precision    recall  f1-score   support")
            print()
            
            # 各类别行
            for class_name in sorted(class_metrics.keys()):
                metrics = class_metrics[class_name]
                class_num = class_name.replace('class_', '')
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1_score = metrics.get('f1-score', 0)
                support = metrics.get('support', 0)
                
                print(f"     Class {class_num}     {precision:.4f}    {recall:.4f}    {f1_score:.4f}      {support}")
            
            print()
            
            # 计算 accuracy
            total_correct = 0
            total_samples = 0
            for metrics in class_metrics.values():
                # accuracy 计算需要所有类别的正确预测数
                recall = metrics.get('recall', 0)
                support = metrics.get('support', 0)
                total_correct += recall * support
                total_samples += support
            
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            print(f"    accuracy                         {accuracy:.4f}      {total_samples}")
            
            # macro avg 和 weighted avg
            if result.get('macro_avg_precision') is not None:
                print(f"   macro avg     {result.get('macro_avg_precision', 0):.4f}    {result.get('macro_avg_recall', 0):.4f}    {result.get('macro_avg_f1', 0):.4f}      {result.get('macro_avg_support', 0)}")
            
            if result.get('weighted_avg_precision') is not None:
                print(f"weighted avg     {result.get('weighted_avg_precision', 0):.4f}    {result.get('weighted_avg_recall', 0):.4f}    {result.get('weighted_avg_f1', 0):.4f}      {result.get('weighted_avg_support', 0)}")
        
        print()
        print("-" * 70)

    @staticmethod
    def _format_txt_report(result: Dict[str, Any]) -> str:
        """格式化为文本报告 - 恢复原始简洁格式，方便人类阅读"""
        lines = []
        
        # 基本信息 - 简洁格式
        lines.append(f"Dataset: {result.get('Dataset', 'N/A')}")
        
        # 从输入文件路径中提取模型名称
        input_file = result.get('Input file', '')
        if '@@' in input_file:
            model_part = input_file.split('@@')[0].split('/')[-1] if input_file else 'N/A'
        else:
            model_part = 'N/A'
        lines.append(f"Model: {model_part}")
        
        lines.append(f"Evaluation Mode: {result.get('Evaluation mode', 'N/A')}")
        lines.append(f"Total samples: {result.get('Total samples', 0)}")
        lines.append(f"Responses with wrong sample size: {result.get('Responses with wrong sample size', 0)}")
        
        # AUC Score
        auc_score = result.get('AUC Score')
        if auc_score is not None and isinstance(auc_score, (int, float)):
            lines.append(f"AUC Score: {auc_score:.4f}")
        else:
            lines.append(f"AUC Score: {auc_score}")
        
        lines.append("-" * 70)
        lines.append("")
        
        # 重新构建分类报告的原始格式
        class_metrics = result.get('class_metrics', {})
        if class_metrics:
            # 表头
            lines.append("              precision    recall  f1-score   support")
            lines.append("")
            
            # 各类别行
            for class_name in sorted(class_metrics.keys()):
                metrics = class_metrics[class_name]
                class_num = class_name.replace('class_', '')
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1_score = metrics.get('f1-score', 0)
                support = metrics.get('support', 0)
                
                lines.append(f"     Class {class_num}     {precision:.4f}    {recall:.4f}    {f1_score:.4f}      {support}")
            
            lines.append("")
            
            # 计算 accuracy
            total_correct = 0
            total_samples = 0
            for metrics in class_metrics.values():
                # accuracy 计算需要所有类别的正确预测数
                recall = metrics.get('recall', 0)
                support = metrics.get('support', 0)
                total_correct += recall * support
                total_samples += support
            
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            lines.append(f"    accuracy                         {accuracy:.4f}      {total_samples}")
            
            # macro avg 和 weighted avg
            if result.get('macro_avg_precision') is not None:
                lines.append(f"   macro avg     {result.get('macro_avg_precision', 0):.4f}    {result.get('macro_avg_recall', 0):.4f}    {result.get('macro_avg_f1', 0):.4f}      {result.get('macro_avg_support', 0)}")
            
            if result.get('weighted_avg_precision') is not None:
                lines.append(f"weighted avg     {result.get('weighted_avg_precision', 0):.4f}    {result.get('weighted_avg_recall', 0):.4f}    {result.get('weighted_avg_f1', 0):.4f}      {result.get('weighted_avg_support', 0)}")
        
        lines.append("")
        
        return "\n".join(lines)
        """格式化为文本报告 - 适配新的JSON格式"""
        lines = []
        
        # 标题
        lines.append("="*60)
        lines.append("🏆 五层智能架构评估报告")
        lines.append("="*60)
        
        # 基本信息
        lines.append(f"INFO: 数据集: {result.get('Dataset', 'N/A')}")
        lines.append(f"� 输入文件: {result.get('Input file', 'N/A')}")
        lines.append(f"📈 总样本数: {result.get('Total samples', 0)}")
        lines.append(f"�️  评估模式: {result.get('Evaluation mode', 'N/A')}")
        
        # AUC Score
        auc_score = result.get('AUC Score')
        if auc_score is not None:
            if isinstance(auc_score, (int, float)):
                lines.append(f"� AUC Score: {auc_score:.4f}")
            else:
                lines.append(f"TARGET: AUC Score: {auc_score}")
        
        # 类别指标
        class_metrics = result.get('class_metrics', {})
        if class_metrics:
            lines.append("")
            lines.append("📈 分类详细指标:")
            for class_name, metrics in class_metrics.items():
                class_num = class_name.replace('class_', '')
                lines.append(f"   Class {class_num}:")
                lines.append(f"     Precision: {metrics.get('precision', 0):.4f}")
                lines.append(f"     Recall: {metrics.get('recall', 0):.4f}")
                lines.append(f"     F1-Score: {metrics.get('f1-score', 0):.4f}")
                lines.append(f"     Support: {metrics.get('support', 0)}")
        
        # Macro 平均
        if result.get('macro_avg_precision') is not None:
            lines.append("")
            lines.append("INFO: Macro 平均:")
            lines.append(f"   Precision: {result.get('macro_avg_precision', 0):.4f}")
            lines.append(f"   Recall: {result.get('macro_avg_recall', 0):.4f}")
            lines.append(f"   F1-Score: {result.get('macro_avg_f1', 0):.4f}")
            lines.append(f"   Support: {result.get('macro_avg_support', 0)}")
        
        # Weighted 平均
        if result.get('weighted_avg_precision') is not None:
            lines.append("")
            lines.append("� Weighted 平均:")
            lines.append(f"   Precision: {result.get('weighted_avg_precision', 0):.4f}")
            lines.append(f"   Recall: {result.get('weighted_avg_recall', 0):.4f}")
            lines.append(f"   F1-Score: {result.get('weighted_avg_f1', 0):.4f}")
            lines.append(f"   Support: {result.get('weighted_avg_support', 0)}")
        
        lines.append("")
        lines.append("="*60)
        
        return "\n".join(lines)
