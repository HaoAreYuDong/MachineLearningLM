#!/usr/bin/env python3
"""
Machine Learning Model Evaluation Tool - Smart Five-Layer Architecture

Adopts an intelligent five-layer separation architecture with clear responsibilities and smart handling of parsing failures:
Layer 1: File Discovery Layer - Intelligently discover target files based on parameters
Layer 2: Label Statistics Layer - Analyze label distribution and determine smart default values
Layer 3: Single File Evaluation Layer - Independently evaluate each file, marking real/default predictions
Layer 4: Smart Voting Aggregation Layer - Single file passthrough or multi-file smart voting (using only real predictions)
Layer 5: Output Layer - Generate final output files

Core Features:
- Smart error handling: Use statistically optimal default labels when JSON parsing fails
- Prediction source marking: Distinguish between real predictions and default fills
- Smart voting: Only use real predictions for multi-file voting, avoiding default value contamination
- Statistical transparency: Detailed recording of real prediction counts, default prediction counts, voting statistics, etc.

Architecture Advantages:
- Clear layering: Single responsibility per layer, clear logic
- Smart processing: Automatically handle various exception scenarios
- Statistical accuracy: Avoid default values affecting real voting results
- 易于维护：层间解耦，便于独立测试和修改

使用示例：
    # 单文件评估
    python evaluator.py --input_dir /path/to/file.jsonl --output_dir result
    
    # 多文件智能投票评估
    python evaluator.py --input_dir /path/to/directory --output_dir result \
                        --dataset_name mnist --row_shuffle_seeds 42 123 \
                        --train_chunk_size 1000 --test_chunk_size 200 \
                        --model_name bert-base --weighted true

作者: TableSense Team
版本: 5.0 (智能五层架构版)
日期: 2025-08-31
"""

import argparse
import sys

from eval_app import create_app


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(
        description='机器学习分类评估工具 - 统一架构版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  单文件评估:
    %(prog)s --input_dir predictions.jsonl --output_dir results
    
  批量投票评估:
    %(prog)s --input_dir ./predictions --output_dir ./results \\
             --dataset_name dataset_name --row_shuffle_seeds 42 123 \\
             --train_chunk_size 1000 --test_chunk_size 200 \\
             --model_name model_name --weighted true \\
             --split_seed 42
        """
    )
    
    # 运行模式选择 - 统一使用 input_dir 参数
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入路径：单文件模式时传入*.jsonl文件路径，批量模式时传入预测文件目录路径')
    
    # 输出路径参数 - 统一使用 output_dir 参数
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出路径：若以.json/.txt结尾则作为结果文件路径；否则作为目录路径（不存在时自动创建多层目录），在其中自动生成.json和.txt文件')
    parser.add_argument('--dataset_name', type=str, 
                       help='数据集名称')
    parser.add_argument('--split_seed', type=int, default=42,
                       help='数据分割种子 (默认: 42)')
    parser.add_argument('--row_shuffle_seeds', type=int, nargs='+', 
                       help='数据行打乱种子列表')
    parser.add_argument('--train_chunk_size', type=int, 
                       help='训练集大小')
    parser.add_argument('--test_chunk_size', type=int, 
                       help='测试集大小')
    parser.add_argument('--model_name', type=str, 
                       help='模型名称')
    
    # Prompt generation specific options
    parser.add_argument("--weighted", type=lambda v: v.lower() in ("true", "1", "yes", "y"), default=True, required=False,
                        help="使用概率加权投票 (default: True)")

    
    return parser.parse_args()


def main():
    """主函数"""
    try:
        args = setup_args()
        
        # 使用新的分层架构应用
        app = create_app(args)
        app.run()
        
    except Exception as e:
        print(f"ERROR: 评估过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
