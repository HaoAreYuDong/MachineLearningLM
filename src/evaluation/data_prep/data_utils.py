"""
Extensions for data preparation pipeline.

This module contains helper functions that implement the smaller, "optional"
pieces of logic (column shuffling, label transform mapping, adding a pseudo
class, discretization, and saving the label transform metadata).

The main logic (process and split) imports and calls these functions so it's
easy to add/replace extensions later.
"""

import os
import json
import random
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def build_label_mapping(all_original_labels: pd.Series, label_transform_mode: str) -> Tuple[Optional[Dict[Any, int]], List[Any]]:
    """
    Build a label mapping given the global set of labels and a mode.

    Modes supported:
    - "from_zero": map labels to 0..(n-1)
    - "none": return None (no mapping)

    Returns (label_mapping, unique_labels_sorted)
    """
    unique_labels = sorted(all_original_labels.unique())

    if label_transform_mode == "from_zero":
        label_mapping = {old_label: idx for idx, old_label in enumerate(unique_labels)}
    else:
        label_mapping = None

    return label_mapping, unique_labels


def apply_label_mapping(y_train_raw: pd.Series, y_test_raw: pd.Series, label_mapping: Optional[Dict[Any, int]]) -> Tuple[pd.Series, pd.Series]:
    """
    Apply label mapping to train and test label series. If label_mapping is
    None then the original labels are returned after resetting the index.
    """
    if label_mapping is not None:
        y_train = y_train_raw.map(label_mapping).reset_index(drop=True)
        y_test = y_test_raw.map(label_mapping).reset_index(drop=True)
    else:
        y_train = y_train_raw.reset_index(drop=True)
        y_test = y_test_raw.reset_index(drop=True)
    return y_train, y_test


def compute_pseudo_label(unique_labels: List[Any], label_mapping: Optional[Dict[Any, int]]) -> int:
    """
    Compute the pseudo label value based on either the label_mapping (preferred)
    or the original unique labels. Mirrors prior behaviour from the original
    script.
    """
    if label_mapping is not None:
        max_label = max(label_mapping.values())
    else:
        # Fall back to the maximum of the original labels (may raise if not comparable)
        max_label = max(unique_labels)
    return int(max_label) + 1


def add_pseudo_class(X_train: pd.DataFrame, y_train: pd.Series, pseudo_label: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Add a pseudo class row consisting of zeros in the feature columns and the
    provided pseudo_label. Returns (X_train_new, y_train_new).
    """
    num_features = X_train.shape[1]
    pseudo_X = pd.DataFrame(np.zeros((1, num_features)), columns=X_train.columns)
    pseudo_y = pd.Series([pseudo_label], name=y_train.name or "label")

    X_train_new = pd.concat([X_train, pseudo_X], ignore_index=True)
    y_train_new = pd.concat([y_train, pseudo_y], ignore_index=True)

    return X_train_new, y_train_new


def discretize_labels(y_train: pd.Series, y_test: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Convert arbitrary labels into contiguous integer labels using sklearn's
    LabelEncoder. Fits on the union of train+test labels for consistency.
    """
    le = LabelEncoder()
    all_labels = pd.concat([y_train, y_test])
    le.fit(all_labels)
    y_train_new = pd.Series(le.transform(y_train), name=y_train.name)
    y_test_new = pd.Series(le.transform(y_test), name=y_test.name)
    return y_train_new, y_test_new


def col_shuffle(X_train: pd.DataFrame, X_test: pd.DataFrame, random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Shuffle feature columns in a consistent way based on random_seed.
    Returns shuffled (X_train, X_test).
    """
    random.seed(random_seed)
    feature_names = list(X_train.columns)
    feature_indices = list(range(len(feature_names)))
    random.shuffle(feature_indices)

    X_train_shuffled = X_train.iloc[:, feature_indices]
    X_test_shuffled = X_test.iloc[:, feature_indices]

    return X_train_shuffled, X_test_shuffled


def save_label_transform_info(output_dir: str,
                              label_transform_mode: str,
                              add_pseudo_class_flag: bool,
                              unique_labels: List[Any],
                              label_mapping: Optional[Dict[Any, int]],
                              pseudo_label: Optional[int],
                              filename: str = "label_transform_info.json") -> str:
    """
    Save essential label transform information (mapping and mode) to a JSON
    file under output_dir and return the path to the saved file. We intentionally
    limit the stored fields to keep the file compact and avoid unrelated
    variables being persisted.
    """
    mapping_info = {
        "transform_mode": label_transform_mode,
        # Only persist the mapping (if present). Keep types simple (str->int)
        "label_mapping": {str(k): int(v) for k, v in label_mapping.items()} if label_mapping else None,
    }

    os.makedirs(output_dir, exist_ok=True)
    mapping_path = os.path.join(output_dir, filename)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_info, f, ensure_ascii=False, indent=2)

    return mapping_path


# A dictionary of default extension hooks (useful for discoverability)
DEFAULT_EXTENSIONS = {
    "build_label_mapping": build_label_mapping,
    "apply_label_mapping": apply_label_mapping,
    "compute_pseudo_label": compute_pseudo_label,
    "add_pseudo_class": add_pseudo_class,
    "discretize_labels": discretize_labels,
    "col_shuffle": col_shuffle,
    "save_label_transform_info": save_label_transform_info,
}
