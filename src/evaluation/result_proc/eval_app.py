#!/usr/bin/env python3
"""
Machine Learning Evaluator - Main Control            # Layer 2: Label Statistics and Default Value Determination
            print("📊 Layer 2: Label Statistics...")
            label_analysis = LabelStatisticsLayer.compute_default_labels(file_paths)
            print("✅ Completed label statistics and default value determination")
            
            # Layer 3: Tagged Single File Evaluation
            print("📊 Layer 3: Single File Evaluation (Tagged)...")
            file_results = TaggedSingleFileEvaluationLayer.evaluate_files(file_paths, label_analysis['global_label_stats'])
            print(f"✅ Completed evaluation of {len(file_results)} files")

Four-Layer Architecture Design:
Layer 1: File Discovery Layer - Parse parameters and discover target files
Layer 2: Single File Evaluation Layer - Independent evaluation of each file  
Layer 3: Result Aggregation Layer - Multi-file voting aggregation or single-file pass-through
Layer 4: Output Layer - Generate final output files

Author: LM_ML
Version: 4.0 (Layered Architecture Edition)
Date: 2025-08-31
"""

from eval_layer_1 import FileDiscoveryLayer
from eval_layer_2 import LabelStatisticsLayer
from eval_layer_3 import TaggedSingleFileEvaluationLayer
from eval_layer_4 import SmartVotingAggregationLayer
from eval_layer_5 import OutputLayer


class EvaluationApp:
    
    def __init__(self, args):
        self.args = args
        
    def run(self):
        print("🎯 ML Evaluator v5.0")
        print("=" * 60)
        
        try:
            # Layer 1: File Discovery
            print("🔍 Layer 1: File Discovery...")
            file_paths = FileDiscoveryLayer.discover_files(self.args)
            print(f"📁 Discovered {len(file_paths)} files")
            
            # Layer 2: Label Statistics and Default Value Determination
            print("📊 Layer 2: Label Statistics...")
            label_analysis = LabelStatisticsLayer.compute_default_labels(file_paths)
            print("✅ Completed label statistics and default value determination")
            
            # Layer 3: Tagged Single File Evaluation
            print("📊 Layer 3: Single File Evaluation (Tagged)...")
            file_results = TaggedSingleFileEvaluationLayer.evaluate_files(file_paths, label_analysis['global_label_stats'])
            print(f"✅ Completed evaluation of {len(file_results)} files")
            print(f"✅ Completed evaluation of {len(file_results)} files")  # Duplicate kept as in original
            
            # Layer 4: Smart Voting Aggregation
            print("🗳️  Layer 4: Smart Voting Aggregation...")
            final_result = SmartVotingAggregationLayer.aggregate_results(
                file_results, self.args.weighted if hasattr(self.args, 'weighted') else True
            )
            print("✅ Results aggregation completed")
            
            # Layer 5: Output Layer
            print("💾 Layer 5: Result Output...")
            
            # Add parameter information to final_result for use in file naming
            final_result['train_chunk_size'] = getattr(self.args, 'train_chunk_size', 'unknown')
            final_result['test_chunk_size'] = getattr(self.args, 'test_chunk_size', 'unknown')
            final_result['split_seed'] = getattr(self.args, 'split_seed', 'unknown')
            
            OutputLayer.format_and_output(
                final_result, 
                self.args.output_dir, 
                self.args.dataset_name, 
                self.args.model_name, 
                self.args.row_shuffle_seeds
            )
            print("✅ Results saved")
            
            print("🎉 Evaluation completed!")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise

def create_app(args):
    return EvaluationApp(args)
