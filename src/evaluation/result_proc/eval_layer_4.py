#!/usr/bin/env python3
"""
Layer 4: Smart Voting Aggregation Layer

Responsibilities:
1. Receive evaluation results from multiple files (with tags)
2. If only 1 file: directly pass through the result
3. If multiple files: perform smart voting aggregation
4. Only use real predictions for voting, ignore default-filled values
5. Return final result in unified format

Input: Dict[str, dict] - {file name: evaluation result}
Output: dict - Final aggregated result
"""

import os
import logging
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

from sklearn.metrics import classification_report, roc_auc_score



class SmartVotingAggregationLayer:
    """Smart Voting Aggregation Layer - Handles single file pass-through or multi-file smart voting aggregation"""
    
    @staticmethod
    def aggregate_results(file_results: Dict[str, dict], weighted: bool = True) -> dict:
        """
        Intelligently aggregate file evaluation results
        
        Args:
            file_results: {file name: evaluation result}
            weighted: Whether to use weighted voting
            
        Returns:
            dict: Final aggregated result
        """
        # Filter out valid results (with data)
        successful_results = {path: result for path, result in file_results.items() 
                            if result.get('total_samples', 0) > 0}
        
        if not successful_results:
            raise Exception("No valid evaluation results to aggregate")
        
        if len(successful_results) == 1:
            # Single file mode: direct pass-through
            return SmartVotingAggregationLayer._handle_single_file(successful_results)
        else:
            # Multi-file mode: smart voting aggregation
            return SmartVotingAggregationLayer._handle_multiple_files(successful_results, weighted)
    
    @staticmethod
    def _handle_single_file(successful_results: Dict[str, dict]) -> dict:
        """
        Handle single file result (direct pass-through)
        
        Args:
            successful_results: Dictionary containing a single successful result
            
        Returns:
            dict: Pass-through result with aggregation identifier
        """
        file_path, result = next(iter(successful_results.items()))
        
        # Add aggregation information
        final_result = result.copy()
        final_result.update({
            'aggregation_mode': 'single_file',
            'file_count': 1,
            'source_files': [file_path],
            'dataset_name': SmartVotingAggregationLayer._extract_dataset_name(file_path)
        })
        
        print(f"📄 Single file mode: {os.path.basename(file_path)}")
        return final_result
    
    @staticmethod
    def _handle_multiple_files(successful_results: Dict[str, dict], weighted: bool) -> dict:
        """
        Handle multi-file results (smart voting aggregation)
        
        Args:
            successful_results: Dictionary of multiple successful results
            weighted: Whether to use weighted voting
            
        Returns:
            dict: Result after smart voting aggregation
        """
        print(f"🗳️  Multi-file smart voting mode: {len(successful_results)} files")
        
        # Collect data and tags from all files
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
            # Get current file data
            predictions = result.get('y_pred', [])
            ground_truth = result.get('y_true', [])
            prediction_tags = result.get('tags', [])
            file_probs = result.get('y_probs', [])  # Try to get probability information
            
            # If no probability info, fill with 0.5
            if not file_probs or len(file_probs) != len(predictions):
                logger.warning(f"File {result.get('file_path', '')} missing probability info, filling with 0.5")
                auc_probs = [0.5] * len(predictions)
            else:
                auc_probs = file_probs
            
            if all_ground_truth is None:
                all_ground_truth = ground_truth
            
            # Combine predictions, probabilities and tags
            predictions_with_tags = list(zip(predictions, prediction_tags))
            probabilities_with_tags = list(zip(auc_probs, prediction_tags))
            
            all_predictions_with_tags.append(predictions_with_tags)
            all_probabilities_with_tags.append(probabilities_with_tags)
        
        if not all_predictions_with_tags or all_ground_truth is None:
            raise Exception("Smart voting aggregation missing necessary prediction data")
        
        # Perform smart voting
        final_predictions, final_auc_probs = SmartVotingAggregationLayer._smart_voting(
            all_predictions_with_tags, all_probabilities_with_tags, weighted, voting_stats
        )
        
        # Important: Ensure ground truth length matches final prediction length
        # If voting truncated length, also truncate ground truth
        truncated_ground_truth = all_ground_truth[:len(final_predictions)] if all_ground_truth else []
        
        # Generate final report
        report, auc_score = SmartVotingAggregationLayer._generate_aggregated_report(
            final_predictions, truncated_ground_truth, final_auc_probs
        )
        
        # Build aggregated result
        final_result = {
            'aggregation_mode': 'smart_voting',
            'voting_method': 'weighted' if weighted else 'majority',
            'file_count': len(successful_results),
            'source_files': list(successful_results.keys()),
            'classification_report': report,
            'auc_score': auc_score,
            'y_pred': final_predictions,
            'y_true': truncated_ground_truth,  # Use truncated ground truth
            'total_samples': len(final_predictions),
            'real_predictions': voting_stats['total_real_votes'],
            'default_fillings': voting_stats['total_default_votes'],
            'voting_statistics': voting_stats
        }
        
        # Print voting statistics
        print(f"📊 Voting statistics:")
        print(f"   Total voting positions: {voting_stats['total_positions']}")
        print(f"   Positions with all real votes: {voting_stats['positions_with_all_real']}")
        print(f"   Positions with partial real votes: {voting_stats['positions_with_partial_real']}")
        print(f"   Positions with no real votes: {voting_stats['positions_with_no_real']}")
        print(f"   Total real votes: {voting_stats['total_real_votes']}")
        print(f"   Total default votes: {voting_stats['total_default_votes']}")
        
        return final_result
    
    @staticmethod
    def _smart_voting(predictions_with_tags: List[List[Tuple]], probabilities_with_tags: List[List[Tuple]], 
                     weighted: bool, voting_stats: dict) -> Tuple[List, List]:
        """
        Smart voting: Only use real predictions for voting, implements robust handling
        
        Processing logic:
        1. Intelligently handle prediction lists of different lengths (due to parsing errors)
        2. For each position, only use predictions with tag="parser_success" for voting
        3. If all tags at a position are "default", use default value
        4. Supports both weighted voting and simple majority voting
        
        Args:
            predictions_with_tags: List of [(prediction, tag), ...] for each file
            probabilities_with_tags: List of [(probability, tag), ...] for each file
            weighted: Whether to use weighted voting
            voting_stats: Voting statistics dictionary
            
        Returns:
            Tuple[List, List]: (Final predictions, Final probabilities)
        """
        if not predictions_with_tags:
            return [], []
        
        # Handle different prediction lengths: use shortest length as baseline
        # This is because some files may have skipped lines due to JSON parsing errors
        lengths = [len(preds) for preds in predictions_with_tags]
        length = min(lengths)
        
        # If lengths inconsistent, print warning and truncate to shortest length
        if len(set(lengths)) > 1:
            print(f"⚠️  Warning: Found prediction lists of different lengths {lengths}, using shortest length {length} for voting")
            print(f"    This is usually caused by JSON parsing errors in some files")
            # Truncate all lists to shortest length to ensure each position has predictions
            predictions_with_tags = [preds[:length] for preds in predictions_with_tags]
            probabilities_with_tags = [probs[:length] for probs in probabilities_with_tags]
        
        final_predictions = []
        final_auc_probs = []
        
        voting_stats['total_positions'] = length
        
        # Vote for each position
        for i in range(length):
            # Collect all real predictions (tag="parser_success") at position i
            real_votes = []
            real_probs = []
            default_votes = []
            
            for voter_idx in range(len(predictions_with_tags)):
                pred, tag = predictions_with_tags[voter_idx][i]
                prob, prob_tag = probabilities_with_tags[voter_idx][i]
                
                if tag == "parser_success":
                    # Only real predictions participate in voting
                    real_votes.append(pred)
                    real_probs.append(prob)
                    voting_stats['total_real_votes'] += 1
                else:
                    # Collect default values as fallback
                    default_votes.append(pred)
                    voting_stats['total_default_votes'] += 1
            
            # Decision based on number of real votes
            if len(real_votes) == 0:
                # Case 4: All votes are default (m-n=0), use default value
                if default_votes:
                    # Use most common default value
                    vote_counts = Counter(default_votes)
                    most_common_default = vote_counts.most_common(1)[0][0]
                    final_predictions.append(int(most_common_default))
                else:
                    # Fallback for extreme cases
                    final_predictions.append(0)
                final_auc_probs.append(0.5)  # Default probability
                voting_stats['positions_with_no_real'] += 1
                
            else:
                # Have real votes, perform normal voting (Case 4: use m-n real predictions)
                if weighted and len(real_probs) > 0:
                    # Weighted voting: use probability information
                    final_pred = SmartVotingAggregationLayer._weighted_vote_single_position(
                        real_votes, real_probs
                    )
                else:
                    # Simple majority voting: count votes
                    vote_counts = Counter(real_votes)
                    final_pred = vote_counts.most_common(1)[0][0]
                
                final_predictions.append(int(final_pred))
                
                # Calculate average probability
                avg_prob = sum(real_probs) / len(real_probs) if real_probs else 0.5
                final_auc_probs.append(avg_prob)
                
                # Record voting type
                if len(real_votes) == len(predictions_with_tags):
                    voting_stats['positions_with_all_real'] += 1
                else:
                    voting_stats['positions_with_partial_real'] += 1
        
        return final_predictions, final_auc_probs
    
    @staticmethod
    def _weighted_vote_single_position(votes: List, probs: List) -> int:
        """
        Perform weighted voting for a single position
        
        Args:
            votes: List of votes
            probs: List of probabilities
            
        Returns:
            int: Voting result
        """
        if not votes:
            return 0
        
        # Simplified weighted voting: use probability as weight
        weighted_votes = defaultdict(float)
        
        for vote, prob in zip(votes, probs):
            # Use probability confidence as weight
            confidence = abs(prob - 0.5) * 2  # Map [0,1] to confidence in [0,1]
            weighted_votes[vote] += confidence
        
        # Return vote with highest weight
        if weighted_votes:
            return max(weighted_votes, key=weighted_votes.get)
        else:
            return Counter(votes).most_common(1)[0][0]
    
    @staticmethod
    def _generate_aggregated_report(predictions: list, ground_truth: list, auc_probabilities: list) -> Tuple[str, float]:
        """
        Generate aggregated classification report
        
        Args:
            predictions: Final predictions
            ground_truth: True labels
            auc_probabilities: Probabilities for AUC calculation
            
        Returns:
            Tuple[str, float]: (Classification report, AUC score)
        """
        try:
            # Ensure all labels are integers to avoid type mixing errors
            predictions_int = [int(pred) for pred in predictions]
            ground_truth_int = [int(label) for label in ground_truth]
            
            unique_labels = sorted(set(ground_truth_int))
            
            # Generate classification report
            report = classification_report(
                ground_truth_int, predictions_int, 
                labels=unique_labels, 
                target_names=[f"Class {label}" for label in unique_labels],
                digits=4, zero_division=0
            )
            
            # Calculate AUC (only for binary classification)
            auc_score = None
            if len(unique_labels) == 2:
                try:
                    auc_score = roc_auc_score(ground_truth_int, auc_probabilities)
                except Exception as e:
                    print(f"⚠️ Aggregated AUC calculation failed: {e}")
                    
        except Exception as e:
            print(f"❌ Aggregated report generation failed: {e}")
            report = f"Aggregation evaluation failed: {e}"
            auc_score = None
            
        return report, auc_score
    
    @staticmethod
    def _extract_dataset_name(file_path: str) -> str:
        """
        Extract dataset name from file path
        
        Args:
            file_path: File path
            
        Returns:
            str: Dataset name
        """
        filename = os.path.basename(file_path)
        
        # Try to extract dataset name from filename
        if '@@' in filename:
            parts = filename.split('@@')
            if len(parts) >= 2:
                second_part = parts[1]
                if '_' in second_part:
                    return second_part.split('_')[0]
        
        # Alternative approach
        if '_' in filename:
            return filename.split('_')[0]
        
        return filename.replace('.jsonl', '')