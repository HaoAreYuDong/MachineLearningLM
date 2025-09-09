#!/usr/bin/env python3
"""
Layer 5: Output Layer
"""

import os
import json
from typing import Dict, Any


class OutputLayer:
    """Output Layer - Responsible for formatting and outputting final results"""
    
    @staticmethod
    def format_and_output(result: Dict[str, Any], output_dir: str, dataset_name: str, 
                         model_name: str, row_shuffle_seeds: str) -> Dict[str, str]:
        """
        Format and output final results
        
        Args:
            result: Aggregated evaluation result
            output_dir: Output directory
            dataset_name: Dataset name
            model_name: Model name
            row_shuffle_seeds: Random seeds
            
        Returns:
            Dict[str, str]: Output file path information
        """
        # Process model name: take only the last part of the path, handle special characters
        model_name_clean = model_name or 'auto_detected'
        
        # Handle backend::model format
        if '::' in model_name_clean:
            parts = model_name_clean.split('::', 1)
            backend, actual_model = parts[0], parts[1]
            if backend.lower() == 'openai':
                # For openai::model, use the actual model name
                model_name_clean = actual_model
            else:
                # For other backends, use full format
                model_name_clean = model_name_clean.replace('::', '_')
        
        # Handle HuggingFace format paths (e.g., minzl/toy_3550)
        if '/' in model_name_clean:
            model_name_clean = model_name_clean.split('/')[-1]
        
        # Replace all possible special characters with @ symbol
        safe_model_name = model_name_clean.replace('-', '_').replace('.', '_').replace(':', '@').replace('/', '_')
        
        # Process row_shuffle_seeds, remove brackets and spaces, replace special characters with @
        safe_seeds = str(row_shuffle_seeds).replace('[', '@').replace(']', '@').replace(' ', '').replace(',', '_').replace(':', '@')
        
        # Build full output directory path - changed to dataset_name/model_name hierarchy
        model_output_dir = os.path.join(output_dir, dataset_name, safe_model_name)
        
        # Ensure output directory exists
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Extract train_chunk_size and test_chunk_size from result (if available)
        train_size = result.get('train_chunk_size', 'unknown')
        test_size = result.get('test_chunk_size', 'unknown')
        split_seed = result.get('split_seed', 'unknown')
        
        # Generate output filename - new format: model_name@@dataset_Sseed*_trainsize*_testsize*_seed@*_report.json/txt
        base_filename = f"{safe_model_name}@@{dataset_name}_Sseed{split_seed}_trainsize{train_size}_testsize{test_size}_seed{safe_seeds}_report"
        json_output_path = os.path.join(model_output_dir, f"{base_filename}.json")
        txt_output_path = os.path.join(model_output_dir, f"{base_filename}.txt")
        
        # Format result
        formatted_result = OutputLayer._format_result(result, dataset_name, model_name, row_shuffle_seeds)
        
        # Write JSON file
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_result, f, indent=2, ensure_ascii=False)
        
        # Write TXT file
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write(OutputLayer._format_txt_report(formatted_result))
        
        # Print summary
        OutputLayer._print_summary(formatted_result)
        
        print(f"\n💾 Evaluation results saved to:")
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
        """Format result - Reference b41.json format"""
        # Get source file information
        source_files = result.get('source_files', [])
        input_file = source_files[0] if source_files else "Unknown"
        
        # Process classification report
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
        
        # Parse classification report string to extract class metrics
        if classification_report_str and classification_report_str != 'N/A':
            try:
                lines = classification_report_str.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('Class '):
                        # Parse class line: e.g., "Class 0     0.8903    0.9512    0.9198      7978"
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
                        # Parse macro avg line
                        parts = line.split()
                        if len(parts) >= 5:
                            macro_avg_precision = float(parts[2])
                            macro_avg_recall = float(parts[3])
                            macro_avg_f1 = float(parts[4])
                            macro_avg_support = int(parts[5])
                    elif line.startswith('weighted avg'):
                        # Parse weighted avg line
                        parts = line.split()
                        if len(parts) >= 5:
                            weighted_avg_precision = float(parts[2])
                            weighted_avg_recall = float(parts[3])
                            weighted_avg_f1 = float(parts[4])
                            weighted_avg_support = int(parts[5])
            except Exception as e:
                print(f"⚠️ Error parsing classification report: {e}")
        
        # Handle AUC Score
        auc_score = result.get('auc_score', None)
        auc_message = auc_score if auc_score is not None else "Not calculated (requires binary classification)"
        
        # Generate b41.json-like format
        formatted = {
            "Dataset": dataset_name,
            "Input file": input_file,
            "Total samples": result.get('total_samples', 0),
            "Responses with wrong sample size": 0,  # Assume default 0, can add this stat later
            "Evaluation mode": result.get('aggregation_mode', 'unknown').replace('_', ' ').title(),
            "AUC Score": auc_message,
            "class_metrics": class_metrics
        }
        
        # Add macro and weighted averages (if parsed successfully)
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
        """Print evaluation summary - Matches TXT file format"""
        print("\n" + "-" * 70)
        print("EVALUATION SUMMARY")
        print("-" * 70)
        
        # Basic info - concise format matching TXT
        print(f"Dataset: {result.get('Dataset', 'N/A')}")
        
        # Extract model name from input file path
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
        
        # Reconstruct original classification report format - matches TXT exactly
        class_metrics = result.get('class_metrics', {})
        if class_metrics:
            # Table header
            print("              precision    recall  f1-score   support")
            print()
            
            # Class rows
            for class_name in sorted(class_metrics.keys()):
                metrics = class_metrics[class_name]
                class_num = class_name.replace('class_', '')
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1_score = metrics.get('f1-score', 0)
                support = metrics.get('support', 0)
                
                print(f"     Class {class_num}     {precision:.4f}    {recall:.4f}    {f1_score:.4f}      {support}")
            
            print()
            
            # Calculate accuracy
            total_correct = 0
            total_samples = 0
            for metrics in class_metrics.values():
                # Accuracy calculation requires correct predictions for all classes
                recall = metrics.get('recall', 0)
                support = metrics.get('support', 0)
                total_correct += recall * support
                total_samples += support
            
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            print(f"    accuracy                         {accuracy:.4f}      {total_samples}")
            
            # Macro avg and weighted avg
            if result.get('macro_avg_precision') is not None:
                print(f"   macro avg     {result.get('macro_avg_precision', 0):.4f}    {result.get('macro_avg_recall', 0):.4f}    {result.get('macro_avg_f1', 0):.4f}      {result.get('macro_avg_support', 0)}")
            
            if result.get('weighted_avg_precision') is not None:
                print(f"weighted avg     {result.get('weighted_avg_precision', 0):.4f}    {result.get('weighted_avg_recall', 0):.4f}    {result.get('weighted_avg_f1', 0):.4f}      {result.get('weighted_avg_support', 0)}")
        
        print()
        print("-" * 70)

    @staticmethod
    def _format_txt_report(result: Dict[str, Any]) -> str:
        """Format as text report - Restore original concise format for human readability"""
        lines = []
        
        # Basic info - concise format
        lines.append(f"Dataset: {result.get('Dataset', 'N/A')}")
        
        # Extract model name from input file path
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
        
        # Reconstruct original classification report format
        class_metrics = result.get('class_metrics', {})
        if class_metrics:
            # Table header
            lines.append("              precision    recall  f1-score   support")
            lines.append("")
            
            # Class rows
            for class_name in sorted(class_metrics.keys()):
                metrics = class_metrics[class_name]
                class_num = class_name.replace('class_', '')
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1_score = metrics.get('f1-score', 0)
                support = metrics.get('support', 0)
                
                lines.append(f"     Class {class_num}     {precision:.4f}    {recall:.4f}    {f1_score:.4f}      {support}")
            
            lines.append("")
            
            # Calculate accuracy
            total_correct = 0
            total_samples = 0
            for metrics in class_metrics.values():
                # Accuracy calculation requires correct predictions for all classes
                recall = metrics.get('recall', 0)
                support = metrics.get('support', 0)
                total_correct += recall * support
                total_samples += support
            
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            lines.append(f"    accuracy                         {accuracy:.4f}      {total_samples}")
            
            # Macro avg and weighted avg
            if result.get('macro_avg_precision') is not None:
                lines.append(f"   macro avg     {result.get('macro_avg_precision', 0):.4f}    {result.get('macro_avg_recall', 0):.4f}    {result.get('macro_avg_f1', 0):.4f}      {result.get('macro_avg_support', 0)}")
            
            if result.get('weighted_avg_precision') is not None:
                lines.append(f"weighted avg     {result.get('weighted_avg_precision', 0):.4f}    {result.get('weighted_avg_recall', 0):.4f}    {result.get('weighted_avg_f1', 0):.4f}      {result.get('weighted_avg_support', 0)}")
        
        lines.append("")
        
        return "\n".join(lines)