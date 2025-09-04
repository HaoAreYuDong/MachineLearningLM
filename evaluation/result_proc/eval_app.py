#!/usr/bin/env python3
"""
Machine Learning Evaluator - Main Controller

Four-layer separation architecture design:
Layer 1: File Discovery Layer - Parse parameters and discover target files
Layer 2: Single File Evaluation Layer - Perform independent evaluation on each file  
Layer 3: Result Aggregation Layer - Multi-file voting aggregation or single-file passthrough
Layer 4: Output Layer - Generate final output files

Author: TableSense Team
Version: 4.0 (Layered Architecture)
Date: 2025-08-31
"""

from eval_layer_1 import FileDiscoveryLayer
from eval_layer_2 import LabelStatisticsLayer
from eval_layer_3 import TaggedSingleFileEvaluationLayer
from eval_layer_4 import SmartVotingAggregationLayer
from eval_layer_5 import OutputLayer


class EvaluationApp:
    """Evaluation application main controller - Pure layer calls without business logic"""
    
    def __init__(self, args):
        """Initialize application"""
        self.args = args
        
    def run(self):
        """Execute five-layer evaluation process"""
        print("TARGET: ML Evaluator v5.0 - Smart Five-Layer Architecture")
        print("=" * 60)
        
        try:
            # Layer 1: File Discovery Layer
            print("SEARCH: Layer 1: File discovery...")
            file_paths = FileDiscoveryLayer.discover_files(self.args)
            print(f"OUTPUT: Discovered {len(file_paths)} files")
            
            # Layer 2: Label Statistics and Default Value Determination Layer
            print("INFO: Layer 2: Label statistics...")
            label_analysis = LabelStatisticsLayer.compute_default_labels(file_paths)
            print("SUCCESS: Completed label statistics and default value determination")
            
            # Layer 3: Tagged Single File Evaluation Layer
            print("INFO: Layer 3: Single file evaluation (with tagging)...")
            file_results = TaggedSingleFileEvaluationLayer.evaluate_files(file_paths, label_analysis['global_label_stats'])
            print(f"SUCCESS: Completed evaluation of {len(file_results)} files")

            # Layer 4: Smart Voting Aggregation Layer
            print("VOTE:  Layer 4: Smart voting aggregation...")
            aggregated_results = SmartVotingAggregationLayer.aggregate_results(file_results, self.args)
            print("SUCCESS: Completed smart voting aggregation")
            
            # Layer 5: Output Layer
            print("NOTE: Layer 5: Output generation...")
            OutputLayer.generate_output(aggregated_results, self.args)
            print("SUCCESS: Output generation completed")
            
            print("COMPLETED: Evaluation completed!")
            
        except Exception as e:
            print(f"ERROR: Evaluation failed: {e}")
            raise


def create_app(args):
    """Factory function: Create evaluation application instance"""
    return EvaluationApp(args)
