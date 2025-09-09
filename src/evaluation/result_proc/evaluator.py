#!/usr/bin/env python3
"""
Machine Learning Model Evaluation Tool - Intelligent Five-Layer Architecture Version

Uses an intelligent five-layer separation architecture with clear responsibilities for each layer, intelligently handling parsing failures:
Layer 1: File Discovery Layer - Intelligently discovers target files based on parameters
Layer 2: Label Statistics Layer - Analyzes label distribution, determines intelligent defaults
Layer 3: Single File Evaluation Layer - Independently evaluates each file, tags real/default predictions
Layer 4: Smart Voting Aggregation Layer - Single file pass-through or multi-file intelligent voting (using only real predictions)
Layer 5: Output Layer - Generates final output files

Core Features:
- Intelligent error handling: Uses statistically optimal default labels when JSON parsing fails
- Prediction source tagging: Distinguishes between real predictions and default fillings
- Smart voting: Uses only real predictions for multi-file voting, avoiding default value contamination
- Statistical transparency: Detailed recording of real predictions, default predictions, voting statistics

Architecture Advantages:
- Clear layering: Single responsibility for each layer, clear logic
- Intelligent processing: Automatically handles various exception scenarios
- Statistical accuracy: Prevents default values from affecting real voting results
- Easy maintenance: Decoupled layers, easy independent testing and modification

Usage Examples:
    # Single file evaluation
    python evaluator.py --input_dir /path/to/file.jsonl --output_dir result
    
    # Multi-file intelligent voting evaluation
    python evaluator.py --input_dir /path/to/directory --output_dir result \
                        --dataset_name mnist --row_shuffle_seeds 42 123 \
                        --train_chunk_size 1000 --test_chunk_size 200 \
                        --model_name bert-base --weighted true

Author: LM_ML
Version: 5.0 (Intelligent Five-Layer Architecture Edition)
Date: 2025-08-31
"""

import argparse
import sys

from eval_app import create_app


def setup_args():
    """Configure command line arguments"""
    parser = argparse.ArgumentParser(
        description='Machine Learning Classification Evaluation Tool - Unified Architecture Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  Single file evaluation:
    %(prog)s --input_dir predictions.jsonl --output_dir results
    
  Batch voting evaluation:
    %(prog)s --input_dir ./predictions --output_dir ./results \\
             --dataset_name dataset_name --row_shuffle_seeds 42 123 \\
             --train_chunk_size 1000 --test_chunk_size 200 \\
             --model_name model_name --weighted true \\
             --split_seed 42
        """
    )
    
    # Run mode selection - unified use of input_dir parameter
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input path: *.jsonl file path for single file mode, prediction file directory path for batch mode')
    
    # Output path parameter - unified use of output_dir parameter
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output path: If ending with .json/.txt, used as result file path; otherwise used as directory path (automatically creates multi-level directories if not existing), automatically generating .json and .txt files within')
    parser.add_argument('--dataset_name', type=str, 
                       help='Dataset name')
    parser.add_argument('--split_seed', type=int, default=42,
                       help='Data split seed (default: 42)')
    parser.add_argument('--row_shuffle_seeds', type=int, nargs='+', 
                       help='Data row shuffle seed list')
    parser.add_argument('--train_chunk_size', type=int, 
                       help='Training set size')
    parser.add_argument('--test_chunk_size', type=int, 
                       help='Test set size')
    parser.add_argument('--model_name', type=str, 
                       help='Model name')
    
    # Prompt generation specific options
    parser.add_argument("--weighted", type=lambda v: v.lower() in ("true", "1", "yes", "y"), default=True, required=False,
                        help="Use probability weighted voting (default: True)")

    
    return parser.parse_args()


def main():
    """Main function"""
    try:
        args = setup_args()
        
        # Use new layered architecture application
        app = create_app(args)
        app.run()
        
    except Exception as e:
        print(f"❌ Error occurred during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()