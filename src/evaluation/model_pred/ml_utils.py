import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Please install with: pip install xgboost")


class BaseMLRunner:
    
    def __init__(self, model_name, random_state=42):
        self.model_name = model_name.lower()
        self.random_state = random_state
        
        self.supported_models = {
            'randomforest': RandomForestClassifier,
            'rf': RandomForestClassifier,
            'knn': KNeighborsClassifier,
            'xgboost': XGBClassifier if XGBOOST_AVAILABLE else None,
            'xgb': XGBClassifier if XGBOOST_AVAILABLE else None,
        }
        
        self.supported_models = {k: v for k, v in self.supported_models.items() if v is not None}
        
        if self.model_name not in self.supported_models:
            raise ValueError(f"Unsupported model: {model_name}. "
                           f"Supported models: {list(self.supported_models.keys())}")
    
    def create_model(self):
        ModelClass = self.supported_models[self.model_name]
        
        if self.model_name in ['randomforest', 'rf']:
            return ModelClass(
                n_estimators=30,
                random_state=self.random_state,
                n_jobs=8  
            )
        elif self.model_name in ['xgboost', 'xgb']:
            # Default XGBoost parameters updated per user's specification
            xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "learning_rate": 0.05,
                "max_depth": 4,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "n_estimators": 500,
                "random_state": self.random_state,
                "verbosity": 0,
                "n_jobs": 1,
            }
            return ModelClass(**xgb_params)
        elif self.model_name == 'knn':
            # KNN defaults adjusted per user's specification
            return ModelClass(
                n_neighbors=8,
                weights="distance",
                p=2
            )
        else:
            return ModelClass(random_state=self.random_state)
    
    def load_label_mapping(self, split_dir):
        label_info_path = os.path.join(split_dir, "label_transform_info.json")
        
        if os.path.exists(label_info_path):
            with open(label_info_path, 'r') as f:
                label_info = json.load(f)
            return label_info.get("label_mapping", {})
        else:
            return {}
    
    def create_unified_label_mapping(self, train_labels, test_labels, original_mapping):
        """
        Create a unified label mapping
        
        Args:
            train_labels: All unique labels in the training set
            test_labels: All unique labels in the test set  
            original_mapping: The original label mapping
            
        Returns:
            tuple: (original_to_unified, unified_to_original)
        """
        all_labels = sorted(set(train_labels) | set(test_labels))
        all_labels = [label.item() if hasattr(label, 'item') else label for label in all_labels]
        
        original_to_unified = {label: idx for idx, label in enumerate(all_labels)}
        unified_to_original = {idx: label for idx, label in enumerate(all_labels)}
        
        return original_to_unified, unified_to_original
    
    def validate_column_types(self, X_train, X_test):
        """
        Enhanced data type detection to ensure mixed-type columns are correctly identified as categorical features
        
        Args:
            X_train: Training set features
            X_test: Test set features
            
        Returns:
            tuple: (validated_X_train, validated_X_test, categorical_columns)
        """
        X_train_validated = X_train.copy()
        X_test_validated = X_test.copy()
        categorical_columns = set()
        
        for column in X_train.columns:
            train_values = X_train[column].dropna()
            test_values = X_test[column].dropna()
            
            if X_train[column].dtype in ['object', 'string']:
                categorical_columns.add(column)
                continue
            
            is_categorical = False
            
            try:
                pd.to_numeric(train_values, errors='raise')
            except (ValueError, TypeError):
                is_categorical = True
            
            if not is_categorical:
                try:
                    pd.to_numeric(test_values, errors='raise')
                except (ValueError, TypeError):
                    is_categorical = True
            
            if is_categorical:
                categorical_columns.add(column)
                X_train_validated[column] = X_train_validated[column].astype('object')
                X_test_validated[column] = X_test_validated[column].astype('object')
        
        return X_train_validated, X_test_validated, categorical_columns

    def create_xgboost_label_mapping(self, train_labels, test_labels, original_mapping):
        """
        Create specialized label mapping for XGBoost to ensure labels start from 0 and are continuous
        
        Args:
            train_labels: All unique labels in the training set
            test_labels: All unique labels in the test set  
            original_mapping: The original label mapping
            
        Returns:
            tuple: (original_to_unified, unified_to_original)
        """
        train_labels = [label.item() if hasattr(label, 'item') else label for label in train_labels]
        test_labels = [label.item() if hasattr(label, 'item') else label for label in test_labels]
        
        sorted_train_labels = sorted(set(train_labels))
        original_to_unified = {}
        unified_to_original = {}
        
        for idx, label in enumerate(sorted_train_labels):
            original_to_unified[label] = idx
            unified_to_original[idx] = label
        
        next_idx = len(sorted_train_labels)
        for label in sorted(set(test_labels)):
            if label not in original_to_unified:
                original_to_unified[label] = next_idx
                unified_to_original[next_idx] = label
                next_idx += 1
        
        return original_to_unified, unified_to_original
    
    def encode_features(self, X_train, X_test):
        """
        Encode features to handle text/categorical features
        
        Args:
            X_train: Training set features
            X_test: Test set features
            
        Returns:
            tuple: (X_train_encoded, X_test_encoded)
        """
        X_train_validated, X_test_validated, categorical_columns = self.validate_column_types(X_train, X_test)
        
        X_train_encoded = X_train_validated.copy()
        X_test_encoded = X_test_validated.copy()
        
        for column in X_train_validated.columns:
            if column in categorical_columns or X_train_validated[column].dtype in ['object', 'string']:
                le = LabelEncoder()
                
                combined_values = pd.concat([X_train_validated[column], X_test_validated[column]], ignore_index=True)
                combined_values = combined_values.fillna('missing') 
                
                le.fit(combined_values.astype(str))
                
                X_train_encoded[column] = le.transform(X_train_validated[column].fillna('missing').astype(str))
                X_test_encoded[column] = le.transform(X_test_validated[column].fillna('missing').astype(str))
                
            elif X_train_validated[column].dtype in ['float64', 'int64']:
                X_train_encoded[column] = X_train_validated[column].fillna(X_train_validated[column].median())
                X_test_encoded[column] = X_test_validated[column].fillna(X_train_validated[column].median())  
        
        return X_train_encoded, X_test_encoded
    
    def train_and_predict_single(self, train_data, test_data, index, label_mapping):
        try:
            X_train = pd.read_csv(train_data['X_train'])
            y_train = pd.read_csv(train_data['y_train'])['label'].values
            
            X_test = pd.read_csv(test_data['X_test'])
            y_test = pd.read_csv(test_data['y_test'])['label'].values
            
            X_train_encoded, X_test_encoded = self.encode_features(X_train, X_test)
            
            train_unique_labels = [label.item() if hasattr(label, 'item') else label for label in set(y_train)]
            test_unique_labels = [label.item() if hasattr(label, 'item') else label for label in set(y_test)]
            
            if self.model_name in ['xgboost', 'xgb']:
                original_to_unified, unified_to_original = self.create_xgboost_label_mapping(
                    train_unique_labels, test_unique_labels, label_mapping
                )
            else:
                original_to_unified, unified_to_original = self.create_unified_label_mapping(
                    train_unique_labels, test_unique_labels, label_mapping
                )
            
            y_train_mapped = np.array([original_to_unified[label] for label in y_train])
            y_test_mapped = np.array([original_to_unified[label] for label in y_test])
            
            if self.model_name in ['xgboost', 'xgb']:
                train_classes = sorted(set(y_train_mapped))
                expected_classes = list(range(len(unified_to_original)))
                
                if train_classes != expected_classes:
                    print(f"⚠️ Warning: XGBoost training set for index {index} has incomplete classes")
                    print(f"   Expected classes: {expected_classes}, Got: {train_classes}")
                    print(f"   Train labels: {sorted(set(y_train))}, Test labels: {sorted(set(y_test))}")
                    
                    from collections import Counter
                    train_label_counts = Counter(y_train)
                    most_frequent_original_label = train_label_counts.most_common(1)[0][0]
                    
                    print(f"   Using fallback prediction: most frequent training label = {most_frequent_original_label}")
                    
                    y_pred_original = np.full(len(y_test), most_frequent_original_label)
                    
                    predictions = []
                    ground_truths = []
                    
                    for i in range(len(y_test)):
                        pred_label = most_frequent_original_label
                        true_label = y_test[i]
                        
                        if hasattr(pred_label, 'item'):
                            pred_label = pred_label.item()
                        if hasattr(true_label, 'item'):
                            true_label = true_label.item()
                        
                        predictions.append({"id": i, "label": pred_label})
                        ground_truths.append({"id": i, "label": true_label})
                    
                    proba_info = []
                    available_labels = [str(label) for label in sorted(unified_to_original.values())]
                    for i in range(len(y_test)):
                        sample_proba = {
                            "id": str(i),
                            "label_probs": [{"label": str(most_frequent_original_label), "prob": 1.0}]
                        }
                        proba_info.append(sample_proba)
                    
                    fallback_accuracy = np.mean(y_test == most_frequent_original_label)
                    
                    return {
                        "id": index,
                        "response": json.dumps(predictions),
                        "groundtruth": json.dumps(ground_truths),
                        "batch_probabilities": proba_info,
                        "available_labels": available_labels,
                        "accuracy": fallback_accuracy,
                        "label_mapping": {
                            "original_to_unified": original_to_unified,
                            "unified_to_original": unified_to_original
                        }
                    }
            
            model = self.create_model()
            model.fit(X_train_encoded, y_train_mapped)
            
            y_pred_mapped = model.predict(X_test_encoded)
            
            try:
                y_pred_proba = model.predict_proba(X_test_encoded)
                proba_info = []
                for i, probas in enumerate(y_pred_proba):
                    sample_proba = {
                        "id": str(i),
                        "label_probs": []
                    }
                    for class_idx, prob in enumerate(probas):
                        original_label = unified_to_original[class_idx]
                        sample_proba["label_probs"].append({
                            "label": str(original_label),
                            "prob": float(prob)
                        })
                    proba_info.append(sample_proba)
            except:
                proba_info = []
            
            y_pred_original = np.array([unified_to_original[pred] for pred in y_pred_mapped])
            y_test_original = np.array([unified_to_original[true] for true in y_test_mapped])
            
            predictions = []
            ground_truths = []
            
            for i in range(len(y_pred_original)):
                pred_label = y_pred_original[i]
                true_label = y_test_original[i]
                
                if hasattr(pred_label, 'item'):
                    pred_label = pred_label.item()
                if hasattr(true_label, 'item'):
                    true_label = true_label.item()
                
                predictions.append({"id": i, "label": pred_label})
                ground_truths.append({"id": i, "label": true_label})
            
            accuracy = accuracy_score(y_test_mapped, y_pred_mapped)
            
            return {
                "id": index,
                "response": json.dumps(predictions),
                "groundtruth": json.dumps(ground_truths),
                "batch_probabilities": proba_info,
                "available_labels": [str(label) for label in sorted(unified_to_original.values())],
                "accuracy": accuracy,
                "label_mapping": {
                    "original_to_unified": original_to_unified,
                    "unified_to_original": unified_to_original
                }
            }
            
        except Exception as e:
            raise Exception(f"Error processing index {index}: {e}")


def extract_model_prefix(model_name):
    model_name = model_name.replace('-', '_').replace('.', '_')
    return model_name


def construct_file_paths(input_dir, output_dir, dataset_name, split_seed, row_shuffle_seed, 
                        train_chunk_size, test_chunk_size, model_name):
    
    model_prefix = extract_model_prefix(model_name)
    
    subdir = f"{dataset_name}_Sseed{split_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}"
    
    input_path = os.path.join(input_dir, dataset_name, subdir)
    
    output_filename = f"{model_prefix}@@{dataset_name}_Rseed{row_shuffle_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}.jsonl"
    output_jsonl = os.path.join(output_dir, dataset_name, subdir, output_filename)
    
    return input_path, output_jsonl


def determine_input_output_paths(input_dir, output_dir, dataset_name=None, 
                                split_seed=42, row_shuffle_seed=123, train_chunk_size=600, 
                                test_chunk_size=7, model_name=None):
    
    if not dataset_name:
        raise ValueError("dataset_name is required for ML model prediction")
    if not model_name:
        raise ValueError("model_name is required for ML model prediction")
    
    input_path, output_file = construct_file_paths(
        input_dir, output_dir, dataset_name, split_seed, row_shuffle_seed,
        train_chunk_size, test_chunk_size, model_name
    )
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    return input_path, output_file
