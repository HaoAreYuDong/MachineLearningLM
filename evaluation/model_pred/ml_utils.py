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
    print("WARNING:  XGBoost not available. Please install with: pip install xgboost")


class BaseMLRunner:
    """机器学习模型基础类"""
    
    def __init__(self, model_name, random_state=42):
        self.model_name = model_name.lower()
        self.random_state = random_state
        
        # 支持的机器学习算法
        self.supported_models = {
            'randomforest': RandomForestClassifier,
            'rf': RandomForestClassifier,
            'knn': KNeighborsClassifier,
            'xgboost': XGBClassifier if XGBOOST_AVAILABLE else None,
            'xgb': XGBClassifier if XGBOOST_AVAILABLE else None,
        }
        
        # 移除不可用的模型
        self.supported_models = {k: v for k, v in self.supported_models.items() if v is not None}
        
        if self.model_name not in self.supported_models:
            raise ValueError(f"Unsupported model: {model_name}. "
                           f"Supported models: {list(self.supported_models.keys())}")
    
    def create_model(self):
        """创建模型实例"""
        ModelClass = self.supported_models[self.model_name]
        
        if self.model_name in ['randomforest', 'rf']:
            return ModelClass(
                n_estimators=30,
                random_state=self.random_state,
                n_jobs=8  # 单线程，因为我们已经在多进程层面并行了
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
        """加载标签映射信息"""
        label_info_path = os.path.join(split_dir, "label_transform_info.json")
        
        if os.path.exists(label_info_path):
            with open(label_info_path, 'r') as f:
                label_info = json.load(f)
            return label_info.get("label_mapping", {})
        else:
            # 如果没有label_transform_info.json，返回空映射
            return {}
    
    def create_unified_label_mapping(self, train_labels, test_labels, original_mapping):
        """
        创建统一的标签映射
        
        Args:
            train_labels: 训练集中的所有唯一标签
            test_labels: 测试集中的所有唯一标签  
            original_mapping: 原始的标签映射
            
        Returns:
            tuple: (original_to_unified, unified_to_original)
        """
        # 合并所有唯一标签，并转换为Python原生类型
        all_labels = sorted(set(train_labels) | set(test_labels))
        all_labels = [label.item() if hasattr(label, 'item') else label for label in all_labels]
        
        # 创建从原始标签到统一标签(0,1,2,...)的映射
        original_to_unified = {label: idx for idx, label in enumerate(all_labels)}
        unified_to_original = {idx: label for idx, label in enumerate(all_labels)}
        
        return original_to_unified, unified_to_original
    
    def validate_column_types(self, X_train, X_test):
        """
        增强数据类型检测，确保混合类型列被正确识别为类别特征
        
        Args:
            X_train: 训练集特征
            X_test: 测试集特征
            
        Returns:
            tuple: (validated_X_train, validated_X_test, categorical_columns)
        """
        X_train_validated = X_train.copy()
        X_test_validated = X_test.copy()
        categorical_columns = set()
        
        for column in X_train.columns:
            # 检查训练集和测试集中是否有无法转换为数值的值
            train_values = X_train[column].dropna()
            test_values = X_test[column].dropna()
            
            # 如果列已经是object或string类型，直接标记为类别特征
            if X_train[column].dtype in ['object', 'string']:
                categorical_columns.add(column)
                continue
            
            # 对于标识为数值类型的列，尝试转换以检测隐藏的字符串值
            is_categorical = False
            
            # 检查训练集
            try:
                pd.to_numeric(train_values, errors='raise')
            except (ValueError, TypeError):
                is_categorical = True
            
            # 检查测试集
            if not is_categorical:
                try:
                    pd.to_numeric(test_values, errors='raise')
                except (ValueError, TypeError):
                    is_categorical = True
            
            # 如果发现无法转换的值，将整列标记为类别特征
            if is_categorical:
                categorical_columns.add(column)
                # 将列类型改为object以确保后续正确处理
                X_train_validated[column] = X_train_validated[column].astype('object')
                X_test_validated[column] = X_test_validated[column].astype('object')
        
        return X_train_validated, X_test_validated, categorical_columns

    def create_xgboost_label_mapping(self, train_labels, test_labels, original_mapping):
        """
        为XGBoost创建专门的标签映射，确保标签从0开始连续
        
        Args:
            train_labels: 训练集中的所有唯一标签
            test_labels: 测试集中的所有唯一标签  
            original_mapping: 原始的标签映射
            
        Returns:
            tuple: (original_to_unified, unified_to_original)
        """
        # 转换为Python原生类型
        train_labels = [label.item() if hasattr(label, 'item') else label for label in train_labels]
        test_labels = [label.item() if hasattr(label, 'item') else label for label in test_labels]
        
        # 先处理训练集标签，按排序顺序映射到0,1,2...
        sorted_train_labels = sorted(set(train_labels))
        original_to_unified = {}
        unified_to_original = {}
        
        # 训练集标签映射到0,1,2...
        for idx, label in enumerate(sorted_train_labels):
            original_to_unified[label] = idx
            unified_to_original[idx] = label
        
        # 处理测试集中出现但训练集没有的标签
        next_idx = len(sorted_train_labels)
        for label in sorted(set(test_labels)):
            if label not in original_to_unified:
                original_to_unified[label] = next_idx
                unified_to_original[next_idx] = label
                next_idx += 1
        
        return original_to_unified, unified_to_original
    
    def encode_features(self, X_train, X_test):
        """
        对特征进行编码处理，处理文本/类别特征
        
        Args:
            X_train: 训练集特征
            X_test: 测试集特征
            
        Returns:
            tuple: (X_train_encoded, X_test_encoded)
        """
        # 首先进行增强数据类型检测
        X_train_validated, X_test_validated, categorical_columns = self.validate_column_types(X_train, X_test)
        
        X_train_encoded = X_train_validated.copy()
        X_test_encoded = X_test_validated.copy()
        
        # 处理每一列
        for column in X_train_validated.columns:
            # 使用验证结果决定处理方式
            if column in categorical_columns or X_train_validated[column].dtype in ['object', 'string']:
                # 文本/类别特征，使用LabelEncoder
                le = LabelEncoder()
                
                # 合并训练和测试集的唯一值来fit encoder
                combined_values = pd.concat([X_train_validated[column], X_test_validated[column]], ignore_index=True)
                combined_values = combined_values.fillna('missing')  # 处理NaN值
                
                le.fit(combined_values.astype(str))
                
                # 编码训练集和测试集
                X_train_encoded[column] = le.transform(X_train_validated[column].fillna('missing').astype(str))
                X_test_encoded[column] = le.transform(X_test_validated[column].fillna('missing').astype(str))
                
            elif X_train_validated[column].dtype in ['float64', 'int64']:
                # 数值特征，处理NaN值
                X_train_encoded[column] = X_train_validated[column].fillna(X_train_validated[column].median())
                X_test_encoded[column] = X_test_validated[column].fillna(X_train_validated[column].median())  # 用训练集的中位数填充测试集
        
        return X_train_encoded, X_test_encoded
    
    def train_and_predict_single(self, train_data, test_data, index, label_mapping):
        """训练单个模型并进行预测"""
        try:
            # 加载训练数据
            X_train = pd.read_csv(train_data['X_train'])
            y_train = pd.read_csv(train_data['y_train'])['label'].values
            
            # 加载测试数据
            X_test = pd.read_csv(test_data['X_test'])
            y_test = pd.read_csv(test_data['y_test'])['label'].values
            
            # 特征编码处理（处理文本/类别特征）
            X_train_encoded, X_test_encoded = self.encode_features(X_train, X_test)
            
            # 获取训练和测试集中的唯一标签
            train_unique_labels = [label.item() if hasattr(label, 'item') else label for label in set(y_train)]
            test_unique_labels = [label.item() if hasattr(label, 'item') else label for label in set(y_test)]
            
            # 根据模型类型选择不同的标签映射策略
            if self.model_name in ['xgboost', 'xgb']:
                # XGBoost使用专门的映射函数，确保标签从0开始连续
                original_to_unified, unified_to_original = self.create_xgboost_label_mapping(
                    train_unique_labels, test_unique_labels, label_mapping
                )
            else:
                # 其他模型使用通用的标签映射
                original_to_unified, unified_to_original = self.create_unified_label_mapping(
                    train_unique_labels, test_unique_labels, label_mapping
                )
            
            # 映射标签到统一的连续值
            y_train_mapped = np.array([original_to_unified[label] for label in y_train])
            y_test_mapped = np.array([original_to_unified[label] for label in y_test])
            
            # 只对XGBoost检查训练集中的类别是否完整
            if self.model_name in ['xgboost', 'xgb']:
                train_classes = sorted(set(y_train_mapped))
                expected_classes = list(range(len(unified_to_original)))
                
                if train_classes != expected_classes:
                    print(f"WARNING: Warning: XGBoost training set for index {index} has incomplete classes")
                    print(f"   Expected classes: {expected_classes}, Got: {train_classes}")
                    print(f"   Train labels: {sorted(set(y_train))}, Test labels: {sorted(set(y_test))}")
                    
                    # 计算训练集中最高频的类别（原始标签）
                    from collections import Counter
                    train_label_counts = Counter(y_train)
                    most_frequent_original_label = train_label_counts.most_common(1)[0][0]
                    
                    print(f"   Using fallback prediction: most frequent training label = {most_frequent_original_label}")
                    
                    # 所有预测都设为训练集中最高频的类别
                    y_pred_original = np.full(len(y_test), most_frequent_original_label)
                    
                    # 构建预测结果（与正常情况格式保持一致）
                    predictions = []
                    ground_truths = []
                    
                    for i in range(len(y_test)):
                        # 确保标签是Python原生类型
                        pred_label = most_frequent_original_label
                        true_label = y_test[i]
                        
                        # 转换为Python原生类型
                        if hasattr(pred_label, 'item'):
                            pred_label = pred_label.item()
                        if hasattr(true_label, 'item'):
                            true_label = true_label.item()
                        
                        predictions.append({"id": i, "label": pred_label})
                        ground_truths.append({"id": i, "label": true_label})
                    
                    # 构建概率信息（100%概率预测为最高频类别）
                    proba_info = []
                    available_labels = [str(label) for label in sorted(unified_to_original.values())]
                    for i in range(len(y_test)):
                        sample_proba = {
                            "id": str(i),
                            "label_probs": [{"label": str(most_frequent_original_label), "prob": 1.0}]
                        }
                        proba_info.append(sample_proba)
                    
                    # 计算这种fallback策略的准确率（用真实的测试标签）
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
            
            # 创建并训练模型
            model = self.create_model()
            model.fit(X_train_encoded, y_train_mapped)
            
            # 预测
            y_pred_mapped = model.predict(X_test_encoded)
            
            # 获取预测概率
            try:
                y_pred_proba = model.predict_proba(X_test_encoded)
                # 构建概率信息
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
            
            # 将预测结果映射回原始标签
            y_pred_original = np.array([unified_to_original[pred] for pred in y_pred_mapped])
            y_test_original = np.array([unified_to_original[true] for true in y_test_mapped])
            
            # 构建预测结果
            predictions = []
            ground_truths = []
            
            for i in range(len(y_pred_original)):
                # 确保标签是Python原生类型，而不是numpy类型
                pred_label = y_pred_original[i]
                true_label = y_test_original[i]
                
                # 转换为Python原生类型
                if hasattr(pred_label, 'item'):
                    pred_label = pred_label.item()
                if hasattr(true_label, 'item'):
                    true_label = true_label.item()
                
                predictions.append({"id": i, "label": pred_label})
                ground_truths.append({"id": i, "label": true_label})
            
            # 计算准确率
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
    """提取模型名称前缀用于文件命名"""
    # 直接使用模型名称，替换特殊字符
    model_name = model_name.replace('-', '_').replace('.', '_')
    return model_name


def construct_file_paths(input_dir, output_dir, dataset_name, split_seed, row_shuffle_seed, 
                        train_chunk_size, test_chunk_size, model_name):
    """根据参数构造文件路径"""
    
    # 提取模型前缀
    model_prefix = extract_model_prefix(model_name)
    
    # 统一的子目录格式
    subdir = f"{dataset_name}_Sseed{split_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}"
    
    # 输入目录路径（不是文件，而是目录）
    input_path = os.path.join(input_dir, dataset_name, subdir)
    
    # 输出文件路径
    output_filename = f"{model_prefix}@@{dataset_name}_Rseed{row_shuffle_seed}_trainSize{train_chunk_size}_testSize{test_chunk_size}.jsonl"
    output_jsonl = os.path.join(output_dir, dataset_name, subdir, output_filename)
    
    return input_path, output_jsonl


def determine_input_output_paths(input_dir, output_dir, dataset_name=None, 
                                split_seed=42, row_shuffle_seed=123, train_chunk_size=600, 
                                test_chunk_size=7, model_name=None):
    """智能判断输入输出路径"""
    
    if not dataset_name:
        raise ValueError("dataset_name is required for ML model prediction")
    if not model_name:
        raise ValueError("model_name is required for ML model prediction")
    
    # 构建输入和输出路径
    input_path, output_file = construct_file_paths(
        input_dir, output_dir, dataset_name, split_seed, row_shuffle_seed,
        train_chunk_size, test_chunk_size, model_name
    )
    
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    return input_path, output_file
