#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_layer_3.py - Layer 3: Single File Evaluation with Tagging

Handles evaluation logic for individual files, distinguishing between real predictions and default-filled data.
Uses high-frequency labels for filling when parsing fails, and tags each prediction accordingly.

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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TaggedSingleFileEvaluationLayer:
    """Layer 3: Single File Evaluation with Tagging
    
    Responsible for processing evaluation logic for individual files, distinguishing between real predictions and default-filled data,
    and adding tags ('real' for real predictions, 'default' for default-filled).
    """
    
    def __init__(self):
        """Initialize the single file evaluation layer"""
        pass
    
    @staticmethod
    def evaluate_single_file_with_tags(file_path: str, label_stats: Dict[str, int]) -> Dict[str, Any]:
        """Evaluate a single file, distinguishing real predictions from default-filled, and adding tags
        
        Args:
            file_path: Path to the file to evaluate
            label_stats: Label statistics information, used to get high-frequency label as default
            
        Returns:
            Dictionary containing true labels, predicted labels, tags, and metadata
        """
        try:
            logger.info(f"Evaluating single file: {file_path}")
            
            # Get default label (high-frequency label)
            if not label_stats:
                default_label = "0"  # Default to 0
            else:
                default_label = str(max(label_stats.items(), key=lambda x: x[1])[0])
            
            logger.info(f"Starting file evaluation, default label set to: {default_label}")
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"File is empty: {file_path}")
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
            
            # Initialize tracking variables
            y_true = []  # True labels
            y_pred = []  # Predicted labels  
            y_probs = [] # Positive class probability (label=1)
            tags = []    # Parsing status tags
            
            # Statistics counters
            real_count = 0
            default_count = 0
            json_error_count = 0
            
            lines = content.strip().split('\n')
            
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                    
                try:
                    # Main logic: parse JSON
                    data = json.loads(line)
                    line_id = data.get('id', line_num)  # Get line ID to construct sample ID
                    
                    # Parse response and groundtruth
                    try:
                        response_data = json.loads(data['response'])
                        groundtruth_data = json.loads(data['groundtruth'])
                        
                        # Check length consistency
                        if len(response_data) != len(groundtruth_data):
                            logger.warning(f"Line {line_num} response and groundtruth length mismatch, triggering exception handling")
                            raise ValueError("Response and groundtruth length mismatch")
                        
                        # Process each sample individually
                        for idx, (pred_item, truth_item) in enumerate(zip(response_data, groundtruth_data)):
                            sample_id = f"L{line_id}R{idx}"  # Construct sample ID: L{line_id}R{response_id}
                            
                            # Extract true label
                            true_label = str(truth_item.get('label', default_label))
                            
                            # Extract predicted label
                            pred_result = TaggedSingleFileEvaluationLayer._process_single_prediction(
                                pred_item, data, idx, default_label, sample_id
                            )
                            
                            # Unpack results
                            pred_label = pred_result['pred_label']
                            auc_prob = pred_result['auc_prob'] 
                            tag = pred_result['tag']
                            
                            # Add to results
                            y_true.append(true_label)
                            y_pred.append(pred_label)
                            y_probs.append(auc_prob)
                            tags.append(tag)
                            
                            # Update counters
                            if tag == 'parser_success':
                                real_count += 1
                            else:
                                default_count += 1
                                
                    except Exception as e:
                        # Exception case 1: JSON parsed successfully but response processing error
                        logger.warning(f"Line {line_num} response processing failed: {e}, calling recovery functions")
                        
                        # Call independent functions a and b for recovery
                        recovered_samples = TaggedSingleFileEvaluationLayer._handle_response_error(
                            data, line_num, default_label
                        )
                        
                        # Add recovered samples
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
                    # Robustness 1: JSON parsing failed, call independent functions a and b
                    logger.warning(f"Line {line_num} JSON parsing failed: {e}, calling recovery functions")
                    
                    recovered_samples = TaggedSingleFileEvaluationLayer._handle_json_parse_error(
                        line, line_num, default_label
                    )
                    
                    # Add recovered samples
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
            
            # Calculate ratios
            real_ratio = (real_count / total_samples * 100) if total_samples > 0 else 0
            
            logger.info(f"File {file_path} evaluation completed:")
            logger.info(f"  Total samples: {total_samples}")
            logger.info(f"  Real predictions: {real_count} ({real_ratio:.2f}%)")
            logger.info(f"  Default fillings: {default_count}")
            
            # JSON error information
            if json_error_count > 0:
                logger.warning(f"  JSON errors: {json_error_count} lines failed to parse")
            
            # Default filling statistics
            if default_count > 0:
                logger.warning(f"  ⚠️  Default fillings: {default_count}/{total_samples} samples used default label '{default_label}'")
            
            # Generate classification report and AUC
            classification_report_str = None
            auc_score = None
            
            if total_samples > 0:
                try:
                    # Convert to integer labels for sklearn
                    y_true_int = [int(label) for label in y_true]
                    y_pred_int = [int(label) for label in y_pred]
                    
                    # Generate classification report
                    unique_labels = sorted(set(y_true_int))
                    classification_report_str = classification_report(
                        y_true_int, y_pred_int,
                        labels=unique_labels,
                        target_names=[f"Class {label}" for label in unique_labels],
                        digits=4,
                        zero_division=0
                    )
                    
                    # Calculate AUC (only for binary classification)
                    if len(unique_labels) == 2 and 0 in unique_labels and 1 in unique_labels:
                        try:
                            # Correct AUC calculation: use true labels and predicted probabilities
                            # y_probs already contains positive class (label=1) probability
                            auc_score = roc_auc_score(y_true_int, y_probs)
                            logger.info(f"AUC Score: {auc_score:.4f}")
                        except Exception as e:
                            logger.warning(f"AUC calculation failed: {e}")
                            # Fallback: use predicted labels (less accurate)
                            try:
                                auc_score = roc_auc_score(y_true_int, y_pred_int)
                                logger.warning(f"Using predicted labels for AUC (less accurate): {auc_score:.4f}")
                            except:
                                auc_score = None
                except Exception as e:
                    logger.warning(f"Classification report generation failed: {e}")
            
            return {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_probs': y_probs,  # Add probability information
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
            logger.error(f"Error evaluating file {file_path}: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
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
        """Batch evaluate multiple files, adding tags to each file
        
        Args:
            file_paths: List of file paths to evaluate
            label_stats: Label statistics information
            
        Returns:
            Dictionary of evaluation results for each file {file_path: evaluation_result}
        """
        results = {}
        
        for file_path in file_paths:
            result = TaggedSingleFileEvaluationLayer.evaluate_single_file_with_tags(file_path, label_stats)
            results[file_path] = result
        
        return results
    
    @staticmethod
    def _process_single_prediction(pred_item: dict, data: dict, idx: int, default_label: str, sample_id: str) -> dict:
        """
        Process a single prediction item
        
        Args:
            pred_item: Single prediction item from response
            data: Complete line data (contains batch_probabilities)
            idx: Sample index
            default_label: Default label
            sample_id: Sample ID
            
        Returns:
            dict: Dictionary containing id, pred_label, auc_prob, tag
        """
        # Try to get prediction directly from label field
        if 'label' in pred_item:
            try:
                # Validate label is a valid number
                pred_label = str(int(pred_item['label']))
                
                # Get positive class probability
                auc_prob = TaggedSingleFileEvaluationLayer._extract_positive_class_prob(data, idx)
                
                return {
                    'id': sample_id,
                    'pred_label': pred_label,
                    'auc_prob': auc_prob,
                    'tag': 'parser_success'
                }
            except (ValueError, TypeError):
                # Label not valid number, try inference from probabilities
                pass
        
        # Label field invalid or missing, call independent function b
        result = TaggedSingleFileEvaluationLayer._extract_prediction_from_probabilities(
            data, idx, default_label, sample_id
        )
        return result
    
    @staticmethod
    def _extract_positive_class_prob(data: dict, idx: int) -> float:
        """
        Extract positive class (label=1) probability
        
        Args:
            data: Complete line data
            idx: Sample index
            
        Returns:
            float: Positive class probability, default 0.5
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
            
            # Build probability dictionary
            prob_dict = {}
            for lp in label_probs:
                if isinstance(lp, dict) and 'label' in lp and 'prob' in lp:
                    try:
                        label = str(int(lp['label']))  # Validate numeric label
                        prob = float(lp['prob'])
                        prob_dict[label] = prob
                    except (ValueError, TypeError):
                        continue
            
            # Return positive class (label=1) probability
            if '1' in prob_dict:
                return prob_dict['1']
            elif '0' in prob_dict:
                return 1.0 - prob_dict['0']  # For binary classification, 1 - negative class probability
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    @staticmethod
    def _handle_response_error(data: dict, line_num: int, default_label: str) -> List[dict]:
        """
        Exception case 1: JSON parsed successfully but response processing error
        Calls independent functions a and b
        
        Args:
            data: Parsed JSON data
            line_num: Line number
            default_label: Default label
            
        Returns:
            List[dict]: List of recovered samples
        """
        # Call independent function a: extract groundtruth via regex
        groundtruth_data = TaggedSingleFileEvaluationLayer._extract_groundtruth_by_regex(data, line_num)
        
        if not groundtruth_data:
            # If groundtruth can't be extracted, return single default sample
            return [{
                'id': f"L{line_num}R0",
                'true_label': default_label,
                'pred_label': default_label,
                'auc_prob': 0.5,
                'tag': 'parser_false'
            }]
        
        # Call independent function b: recover predictions based on groundtruth and batch_probabilities
        recovered_samples = []
        line_id = data.get('id', line_num)
        
        for idx, truth_item in enumerate(groundtruth_data):
            sample_id = f"L{line_id}R{idx}"
            true_label = str(truth_item.get('label', default_label))
            
            # Call independent function b for each sample
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
        Robustness 1: JSON parsing failed, call independent functions a and b
        
        Args:
            line: Raw line data
            line_num: Line number
            default_label: Default label
            
        Returns:
            List[dict]: List of recovered samples
        """
        # Call independent function a: extract groundtruth via regex
        groundtruth_data = TaggedSingleFileEvaluationLayer._extract_groundtruth_by_regex_from_line(line)
        
        if not groundtruth_data:
            # If groundtruth can't be extracted, return single default sample
            return [{
                'id': f"L{line_num}R0",
                'true_label': default_label,
                'pred_label': default_label,
                'auc_prob': 0.5,
                'tag': 'parser_false'
            }]
        
        # Try to extract batch_probabilities
        batch_probs_data = TaggedSingleFileEvaluationLayer._extract_batch_probabilities_by_regex(line)
        
        # Build pseudo data object for calling independent function b
        fake_data = {
            'id': line_num,
            'batch_probabilities': batch_probs_data
        }
        
        # Call independent function b: recover predictions based on groundtruth
        recovered_samples = []
        
        for idx, truth_item in enumerate(groundtruth_data):
            sample_id = f"L{line_num}R{idx}"
            true_label = str(truth_item.get('label', default_label))
            
            # Call independent function b for each sample
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
        Independent function a: extract groundtruth from parsed data via regex
        
        Args:
            data: Parsed JSON data
            line_num: Line number (for logging)
            
        Returns:
            List[dict]: Groundtruth data list
        """
        try:
            # First try direct access
            if 'groundtruth' in data:
                groundtruth = data['groundtruth']
                if isinstance(groundtruth, str):
                    return json.loads(groundtruth)
                elif isinstance(groundtruth, list):
                    return groundtruth
            
            logger.warning(f"Line {line_num} unable to get groundtruth from data")
            return []
            
        except Exception as e:
            logger.warning(f"Line {line_num} groundtruth parsing failed: {e}")
            return []
    
    @staticmethod
    def _extract_groundtruth_by_regex_from_line(line: str) -> List[dict]:
        """
        Independent function a: extract groundtruth from raw line via regex
        
        Args:
            line: Raw line data
            
        Returns:
            List[dict]: Groundtruth data list
        """
        
        try:
            # Use regex to extract groundtruth field
            groundtruth_pattern = r'"groundtruth":\s*"(\[.*?\])"'
            match = re.search(groundtruth_pattern, line)
            
            if match:
                groundtruth_str = match.group(1).replace('\\"', '"')
                return json.loads(groundtruth_str)
            
            # Try alternative format "groundtruth": [...]
            groundtruth_pattern2 = r'"groundtruth":\s*(\[.*?\])'
            match2 = re.search(groundtruth_pattern2, line, re.DOTALL)
            
            if match2:
                groundtruth_str = match2.group(1)
                return json.loads(groundtruth_str)
            
            return []
            
        except Exception as e:
            logger.debug(f"Regex groundtruth extraction failed: {e}")
            return []
    
    @staticmethod
    def _extract_batch_probabilities_by_regex(line: str) -> List[dict]:
        """
        Extract batch_probabilities from raw line via regex
        
        Args:
            line: Raw line data
            
        Returns:
            List[dict]: batch_probabilities data list
        """
        
        try:
            # Use regex to extract batch_probabilities field
            pattern = r'"batch_probabilities":\s*(\[.*?\])(?=,\s*"[^"]+"|$)'
            match = re.search(pattern, line, re.DOTALL)
            
            if match:
                batch_probs_str = match.group(1)
                return json.loads(batch_probs_str)
            
            return []
            
        except Exception as e:
            logger.debug(f"Regex batch_probabilities extraction failed: {e}")
            return []
    
    @staticmethod
    def _extract_prediction_from_probabilities(data: dict, idx: int, default_label: str, sample_id: str) -> dict:
        """
        Independent function b: parse prediction from batch_probabilities
        
        Args:
            data: Data containing batch_probabilities
            idx: Sample index (optional)
            default_label: Default label
            sample_id: Sample ID
            
        Returns:
            dict: Dictionary containing id, pred_label, auc_prob, tag
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
            
            # Find matching probability item (by idx or id)
            prob_item = None
            
            # If idx provided, try index matching first
            if 0 <= idx < len(batch_probs):
                candidate = batch_probs[idx]
                if isinstance(candidate, dict) and 'id' in candidate:
                    # Verify id matches (considering string/number conversion)
                    try:
                        if str(candidate['id']) == str(idx):
                            prob_item = candidate
                    except:
                        pass
                
                # If index match fails, use item at index position
                if prob_item is None:
                    prob_item = candidate
            
            # If still not found, traverse to find first matching id
            if prob_item is None:
                for item in batch_probs:
                    if isinstance(item, dict) and 'id' in item:
                        try:
                            if str(item['id']) == str(idx):
                                prob_item = item
                                break
                        except:
                            continue
            
            # If still not found, use first available item
            if prob_item is None and len(batch_probs) > 0:
                prob_item = batch_probs[0]
            
            if prob_item is None or not isinstance(prob_item, dict):
                return {
                    'id': sample_id,
                    'pred_label': default_label,
                    'auc_prob': 0.5,
                    'tag': 'parser_false'
                }
            
            # Parse label_probs
            label_probs = prob_item.get('label_probs', [])
            if not isinstance(label_probs, list):
                return {
                    'id': sample_id,
                    'pred_label': default_label,
                    'auc_prob': 0.5,
                    'tag': 'parser_false'
                }
            
            # Find highest probability valid numeric label
            max_prob = -1
            best_label = default_label
            prob_dict = {}
            
            for lp in label_probs:
                if isinstance(lp, dict) and 'label' in lp and 'prob' in lp:
                    try:
                        label = str(int(lp['label']))  # Validate numeric label
                        prob = float(lp['prob'])
                        prob_dict[label] = prob
                        
                        if prob > max_prob:
                            max_prob = prob
                            best_label = label
                    except (ValueError, TypeError):
                        continue
            
            # Calculate positive class probability
            if '1' in prob_dict:
                auc_prob = prob_dict['1']
            elif '0' in prob_dict:
                auc_prob = 1.0 - prob_dict['0']
            else:
                auc_prob = 0.5
            
            # If valid label found
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
            logger.debug(f"Independent function b parsing failed: {e}")
            return {
                'id': sample_id,
                'pred_label': default_label,
                'auc_prob': 0.5,
                'tag': 'parser_false'
            }
    
    # Below are existing functions kept for compatibility
    
    @staticmethod
    def _extract_label_from_probabilities(data: Dict[str, Any], sample_idx: int, default_label: str) -> Tuple[str, bool]:
        """Extract label from batch_probabilities field
        
        Args:
            data: Data containing batch_probabilities
            sample_idx: Sample index
            default_label: Default label
            
        Returns:
            Tuple[str, bool]: (Predicted label, whether extraction was successful)
        """
        try:
            # Check if batch_probabilities field exists
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
            
            # Find highest probability label
            max_prob = -1
            best_label = default_label
            
            for prob_item in label_probs:
                if isinstance(prob_item, dict) and 'label' in prob_item and 'prob' in prob_item:
                    try:
                        label = str(prob_item['label'])
                        prob = float(prob_item['prob'])
                        
                        # Validate if valid numeric label
                        int(label)  # Verify label is numeric
                        
                        if prob > max_prob:
                            max_prob = prob
                            best_label = label
                    except (ValueError, TypeError):
                        # Label not numeric, skip
                        continue
            
            # If valid label found and probability > 0
            if max_prob > 0:
                return best_label, True
            else:
                return default_label, False
                
        except Exception as e:
            logger.debug(f"Probability analysis failed: {e}")
            return default_label, False

    @staticmethod
    def _extract_label_and_prob_from_probabilities(data: Dict[str, Any], sample_idx: int, default_label: str) -> Tuple[str, float, bool]:
        """Extract label and probability from batch_probabilities field
        
        Args:
            data: Data containing batch_probabilities
            sample_idx: Sample index
            default_label: Default label
            
        Returns:
            Tuple[str, float, bool]: (Predicted label, positive class probability, whether extraction was successful)
        """
        try:
            # Check if batch_probabilities field exists
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
            
            # Find highest probability label, record all probabilities
            max_prob = -1
            best_label = default_label
            prob_dict = {}
            
            for prob_item in label_probs:
                if isinstance(prob_item, dict) and 'label' in prob_item and 'prob' in prob_item:
                    try:
                        label = str(prob_item['label'])
                        prob = float(prob_item['prob'])
                        
                        # Validate if valid numeric label
                        int(label)  # Verify label is numeric
                        
                        prob_dict[label] = prob
                        if prob > max_prob:
                            max_prob = prob
                            best_label = label
                    except (ValueError, TypeError):
                        # Label not numeric, skip
                        continue
            
            # For binary classification, return positive class (label=1) probability
            positive_class_prob = prob_dict.get('1', 0.5)
            if len(prob_dict) == 2 and '0' in prob_dict and '1' in prob_dict:
                # Standard binary case, return positive class probability
                auc_prob = positive_class_prob
            else:
                # Other cases, if prediction is positive class use its probability, else use 1 - negative class probability
                if best_label == '1':
                    auc_prob = max_prob
                elif best_label == '0' and '0' in prob_dict:
                    auc_prob = 1.0 - prob_dict['0']
                else:
                    auc_prob = 0.5
            
            # If valid label found and probability > 0, consider inference successful
            if max_prob > 0:
                return best_label, auc_prob, True
            else:
                return default_label, 0.5, False
                
        except Exception as e:
            logger.debug(f"Probability analysis failed: {e}")
            return default_label, 0.5, False

    @staticmethod
    def _smart_recover_from_parsed_data(data: Dict[str, Any], line_num: int, default_label: str) -> List[Tuple[str, str, str, float]]:
        """
        Intelligently recover batch data from parsed JSON data (optimized version, no regex needed)
        
        Strategy:
        1. Directly get true labels from data['groundtruth']
        2. Directly get probability information from data['batch_probabilities']
        3. Perform probability inference for each position, tag as 'real' if successful, 'default' if failed
        4. Avoid duplicate labels, process each id only once
        
        Args:
            data: Parsed JSON data containing groundtruth and batch_probabilities fields
            line_num: Line number (for logging)
            default_label: Default label
            
        Returns:
            List[Tuple[str, str, str, float]]: [(true_label, pred_label, tag, auc_prob), ...]
        """
        recovered_samples = []
        
        try:
            # 1. Directly get groundtruth data
            if 'groundtruth' not in data:
                logger.warning(f"Line {line_num} missing groundtruth field, using single default sample")
                return [(default_label, default_label, 'default', 0.5)]
            
            # Parse groundtruth (could be string or already object)
            groundtruth_data = data['groundtruth']
            if isinstance(groundtruth_data, str):
                groundtruth_data = json.loads(groundtruth_data)
            
            if not isinstance(groundtruth_data, list):
                logger.warning(f"Line {line_num} groundtruth not in list format")
                return [(default_label, default_label, 'default', 0.5)]
            
            # 2. Directly get batch_probabilities data
            batch_probs_data = data.get('batch_probabilities', [])
            if isinstance(batch_probs_data, str):
                batch_probs_data = json.loads(batch_probs_data)
            
            if not isinstance(batch_probs_data, list):
                logger.warning(f"Line {line_num} batch_probabilities not in list format")
                batch_probs_data = []
            
            logger.debug(f"Line {line_num} successfully extracted {len(batch_probs_data)} probability items")
            
            # 3. Process each sample in order
            used_prob_indices = set()  # Track used probability indices to avoid reuse
            real_inference_items = []  # Track real inference items
            default_filling_items = []  # Track default filling items
            
            for idx, truth_item in enumerate(groundtruth_data):
                if not isinstance(truth_item, dict) or 'label' not in truth_item:
                    logger.warning(f"Line {line_num} item {idx} true label format abnormal")
                    sample_id = str(idx)
                    recovered_samples.append((default_label, default_label, 'default', 0.5))
                    default_filling_items.append((line_num, sample_id))
                    continue
                
                true_label = str(truth_item['label'])
                sample_id = str(truth_item.get('id', idx))  # Use id field if available, else index
                
                # 4. Try to find corresponding probability information from batch_probabilities
                pred_label = default_label
                tag = 'default'
                auc_prob = 0.5  # Default probability
                
                # Find first matching unused probability item
                found_match = False
                for prob_idx, prob_item in enumerate(batch_probs_data):
                    if (prob_idx not in used_prob_indices and
                        isinstance(prob_item, dict) and 
                        'id' in prob_item and 
                        str(prob_item['id']) == sample_id):
                        
                        logger.debug(f"Line {line_num} sample {sample_id} found matching probability item at index {prob_idx}")
                        
                        # Try probability inference (use new function to get both label and probability)
                        inferred_label, inferred_prob, inference_success = TaggedSingleFileEvaluationLayer._extract_label_and_prob_from_probabilities(
                            {'batch_probabilities': [prob_item]}, 0, default_label
                        )
                        
                        used_prob_indices.add(prob_idx)  # Mark this probability item as used
                        found_match = True
                        
                        if inference_success:
                            pred_label = inferred_label
                            auc_prob = inferred_prob
                            tag = 'real'
                            real_inference_items.append((line_num, sample_id))
                            logger.debug(f"Line {line_num} sample {sample_id} probability inference successful: {pred_label}")
                        else:
                            logger.debug(f"Line {line_num} sample {sample_id} probability inference failed, using default label")
                        break
                
                if not found_match:
                    logger.debug(f"Line {line_num} sample {sample_id} no matching probability item found")
                
                # Record result
                recovered_samples.append((true_label, pred_label, tag, auc_prob))
                if tag == 'default':
                    default_filling_items.append((line_num, sample_id))
            
            # Detailed log output
            real_count = sum(1 for _, _, tag in recovered_samples if tag == 'real')
            default_count = sum(1 for _, _, tag in recovered_samples if tag == 'default')
            
            logger.info(f"Line {line_num} intelligent recovery: total samples {len(recovered_samples)}, real predictions {real_count}, default fillings {default_count}")
            
            if real_inference_items:
                real_info = ", ".join([f"line{line}:id{sid}" for line, sid in real_inference_items[:10]])
                logger.info(f"  Real inference success: {real_info}{'...' if len(real_inference_items) > 10 else ''}")
            
            if default_filling_items:
                default_info = ", ".join([f"line{line}:id{sid}" for line, sid in default_filling_items[:10]])
                logger.warning(f"  Default filling items: {default_info}{'...' if len(default_filling_items) > 10 else ''}")
            
            return recovered_samples
            
        except Exception as e:
            logger.warning(f"Line {line_num} intelligent recovery failed: {e}, using single default sample")
            return [(default_label, default_label, 'default', 0.5)]

    @staticmethod
    def _smart_recover_batch_data_from_failed_line(line: str, line_num: int, default_label: str) -> List[Tuple[str, str, str]]:
        """
        Intelligently recover batch data from failed line
        
        Strategy:
        1. Get true labels and count from groundtruth field
        2. Extract probability information from batch_probabilities field in order
        3. Perform probability inference for each position, tag as 'real' if successful, 'default' if failed
        4. Avoid duplicate labels, process each id only once
        
        Args:
            line: Raw line data that failed to parse
            line_num: Line number (for logging)
            default_label: Default label
            
        Returns:
            List[Tuple[str, str, str]]: [(true_label, pred_label, tag), ...]
        """
        
        recovered_samples = []
        
        try:
            # 1. Extract groundtruth field
            groundtruth_pattern = r'"groundtruth":\s*"(\[.*?\])"'
            groundtruth_match = re.search(groundtruth_pattern, line)
            
            if not groundtruth_match:
                logger.warning(f"Line {line_num} unable to find groundtruth field, using single default sample")
                return [(default_label, default_label, 'default', 0.5)]
            
            # Parse groundtruth
            groundtruth_str = groundtruth_match.group(1).replace('\\"', '"')
            groundtruth_data = json.loads(groundtruth_str)
            
            if not isinstance(groundtruth_data, list):
                logger.warning(f"Line {line_num} groundtruth not in list format")
                return [(default_label, default_label, 'default', 0.5)]
            
            # 2. Extract batch_probabilities field
            batch_probs_pattern = r'"batch_probabilities":\s*(\[.*?\])(?=,\s*"[^"]+"|$)'
            batch_probs_match = re.search(batch_probs_pattern, line, re.DOTALL)
            
            batch_probs_data = []
            if batch_probs_match:
                try:
                    batch_probs_str = batch_probs_match.group(1)
                    batch_probs_data = json.loads(batch_probs_str)
                    logger.debug(f"Line {line_num} successfully extracted {len(batch_probs_data)} probability items")
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num} batch_probabilities parsing failed: {e}")
                    batch_probs_data = []
            else:
                logger.warning(f"Line {line_num} unable to find batch_probabilities field")
            
            # 3. Process each sample in order
            used_prob_indices = set()  # Track used probability indices
            real_inference_items = []  # Track real inference items
            default_filling_items = []  # Track default filling items
            
            for idx, truth_item in enumerate(groundtruth_data):
                if not isinstance(truth_item, dict) or 'label' not in truth_item:
                    logger.warning(f"Line {line_num} item {idx} true label format abnormal")
                    sample_id = str(idx)
                    recovered_samples.append((default_label, default_label, 'default', 0.5))
                    default_filling_items.append((line_num, sample_id))
                    continue
                
                true_label = str(truth_item['label'])
                sample_id = str(truth_item.get('id', idx))  # Use id field if available, else index
                
                # 4. Try to find corresponding probability information from batch_probabilities
                # Find first unused matching item in order
                pred_label = default_label
                tag = 'default'
                
                # Find first matching unused probability item
                found_match = False
                for prob_idx, prob_item in enumerate(batch_probs_data):
                    if (prob_idx not in used_prob_indices and
                        isinstance(prob_item, dict) and 
                        'id' in prob_item and 
                        str(prob_item['id']) == sample_id):
                        
                        logger.debug(f"Line {line_num} sample {sample_id} found matching probability item at index {prob_idx}")
                        
                        # Try probability inference
                        inferred_label, inference_success = TaggedSingleFileEvaluationLayer._extract_label_from_probability_item(
                            prob_item, default_label
                        )
                        
                        used_prob_indices.add(prob_idx)  # Mark this probability item as used
                        found_match = True
                        
                        if inference_success:
                            pred_label = inferred_label
                            tag = 'real'
                            real_inference_items.append((line_num, sample_id))
                            logger.debug(f"Line {line_num} sample {sample_id} probability inference successful: {pred_label}")
                        else:
                            logger.debug(f"Line {line_num} sample {sample_id} probability inference failed, using default label")
                        break
                
                if not found_match:
                    logger.debug(f"Line {line_num} sample {sample_id} no matching probability item found")
                
                # Record result
                recovered_samples.append((true_label, pred_label, tag))
                if tag == 'default':
                    default_filling_items.append((line_num, sample_id))
            
            # Detailed log output
            real_count = sum(1 for _, _, tag in recovered_samples if tag == 'real')
            default_count = sum(1 for _, _, tag in recovered_samples if tag == 'default')
            
            logger.info(f"Line {line_num} intelligent recovery: total samples {len(recovered_samples)}, real predictions {real_count}, default fillings {default_count}")
            
            if real_inference_items:
                real_info = ", ".join([f"line{line}:id{sid}" for line, sid in real_inference_items[:10]])
                logger.info(f"  Real inference success: {real_info}{'...' if len(real_inference_items) > 10 else ''}")
            
            if default_filling_items:
                default_info = ", ".join([f"line{line}:id{sid}" for line, sid in default_filling_items[:10]])
                logger.warning(f"  Default filling items: {default_info}{'...' if len(default_filling_items) > 10 else ''}")
            
            return recovered_samples
            
        except Exception as e:
            logger.warning(f"Line {line_num} intelligent recovery failed: {e}, using single default sample")
            return [(default_label, default_label, 'default', 0.5)]

    @staticmethod
    def _extract_label_from_probability_item(prob_item: dict, default_label: str) -> Tuple[str, bool]:
        """
        Extract label from single probability item
        
        Args:
            prob_item: Probability item, format like {"id": "0", "label_probs": [{"label": "0", "prob": 0.95}, ...]}
            default_label: Default label
            
        Returns:
            Tuple[str, bool]: (Predicted label, whether extraction was successful)
        """
        try:
            if 'label_probs' not in prob_item:
                return default_label, False
            
            label_probs = prob_item['label_probs']
            if not isinstance(label_probs, list):
                return default_label, False
            
            # Find highest probability label
            max_prob = -1
            best_label = default_label
            
            for prob_entry in label_probs:
                if isinstance(prob_entry, dict) and 'label' in prob_entry and 'prob' in prob_entry:
                    try:
                        label = str(prob_entry['label'])
                        prob = float(prob_entry['prob'])
                        
                        # Validate if valid numeric label
                        int(label)  # Verify label is numeric
                        
                        if prob > max_prob:
                            max_prob = prob
                            best_label = label
                    except (ValueError, TypeError):
                        # Label not numeric, skip
                        continue
            
            # If valid label found and probability > 0, consider inference successful
            # Even if inference result equals default label, it's a successful probability-based inference
            if max_prob > 0:
                return best_label, True
            else:
                return default_label, False
                
        except Exception:
            return default_label, False