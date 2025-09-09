"""
Extension utilities for prompt generation.
Keep small, independent functions here so the main orchestration file can
remain focused on I/O, parallelism and writing outputs.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import json

# tiktoken will be used by the main process; provide a helper to obtain encoder
import tiktoken

def get_encoder(prompt_obj):
    """Return a tiktoken encoder instance for the given encoding name."""
    local_enc = tiktoken.get_encoding("cl100k_base")
    return len(local_enc.encode(str(prompt_obj)))


def normalize_feature_arrays(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Robust feature normalization for mixed-type tabular data.

    Goals / behaviour:
    1. Identify numeric columns robustly (even if they arrive as object with numeric strings).
    2. Coerce numeric-like columns via ``pd.to_numeric(errors='coerce')``.
    3. Ignore columns that are fully non-numeric (kept as-is).
    4. Compute mean/std on *training* numeric values only (NaNs excluded).
    5. Handle zero / near-zero std by flooring at 1e-6.
    6. Clip standardized values to [-1000, 1000], then map to positive int space as historical logic: int((x * 120) + 500).
    7. Leave non-numeric columns untouched.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Full train and test matrices with normalized numeric cols converted to ints, others preserved (object -> left as original representation).
    """

    # Copy to avoid mutating caller dataframes
    X_train = X_train_df.copy()
    X_test = X_test_df.copy()

    # Attempt to coerce every column to numeric to discover numeric-like columns
    train_numeric_coerced = X_train.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    test_numeric_coerced = X_test.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    # A column is considered numeric if after coercion it has at least one non-null AND not all values are NaN.
    numeric_mask = train_numeric_coerced.notna().any(axis=0)
    numeric_cols = [col for col in X_train.columns if numeric_mask[col]]

    if len(numeric_cols) == 0:
        return X_train.values, X_test.values

    # Extract coerced numeric arrays for selected columns
    X_train_num = train_numeric_coerced[numeric_cols].to_numpy(dtype=float)
    X_test_num = test_numeric_coerced[numeric_cols].to_numpy(dtype=float)

    # Compute mean/std using only training data
    mean = np.nanmean(X_train_num, axis=0)
    std = np.nanstd(X_train_num, axis=0, ddof=1)
    # Replace NaN std (all NaN column) with 1 to avoid division issues; then clip
    std = np.where(np.isnan(std), 1.0, std)
    std = np.clip(std, 1e-6, None)

    # Replace NaNs in train/test with training mean (simple imputation)
    # (If mean is NaN because the entire column was NaN, fallback to 0)
    mean_no_nan = np.where(np.isnan(mean), 0.0, mean)
    train_nan_mask = np.isnan(X_train_num)
    test_nan_mask = np.isnan(X_test_num)
    if train_nan_mask.any():
        X_train_num[train_nan_mask] = np.take(mean_no_nan, np.where(train_nan_mask)[1])
    if test_nan_mask.any():
        X_test_num[test_nan_mask] = np.take(mean_no_nan, np.where(test_nan_mask)[1])

    try:
        X_train_std = (X_train_num - mean_no_nan) / std
        X_test_std = (X_test_num - mean_no_nan) / std
    except TypeError:
        # Fallback: force conversion then retry (extreme edge case if objects slipped in)
        X_train_std = (X_train_num.astype(float) - mean_no_nan) / std
        X_test_std = (X_test_num.astype(float) - mean_no_nan) / std

    X_train_std = np.clip(X_train_std, -1000, 1000)
    X_test_std = np.clip(X_test_std, -1000, 1000)

    # Map to historical positive integer space
    X_train_scaled = ((X_train_std * 120.0) + 500.0)
    X_test_scaled = ((X_test_std * 120.0) + 500.0)

    # Ensure non-negative and cast
    X_train_scaled = np.clip(X_train_scaled, 0, None).astype(int)
    X_test_scaled = np.clip(X_test_scaled, 0, None).astype(int)

    # Write back into copies
    X_train.loc[:, numeric_cols] = X_train_scaled
    X_test.loc[:, numeric_cols] = X_test_scaled

    return X_train.values, X_test.values


def cast_int_non_normalized(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    When normalization is disabled, cast numeric columns to int for consistency
    with previous behaviour.
    """
    X_train_processed = X_train_df.copy()
    X_test_processed = X_test_df.copy()

    numerical_cols = X_train_df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        X_train_processed[numerical_cols] = X_train_processed[numerical_cols].astype(int)
        X_test_processed[numerical_cols] = X_test_processed[numerical_cols].astype(int)

    return X_train_processed.values, X_test_processed.values


def format_lines(
    X_train_arr: np.ndarray,
    y_train_arr: np.ndarray,
    X_test_arr: np.ndarray,
    feature_names: List[str],
    prompt_format_style: str = "concat",
) -> Tuple[List[str], List[str]]:
    """
    Convert numeric/feature arrays into the pair of lists (train_lines, test_lines)
    according to the requested prompt_format_style.
    """
    if prompt_format_style == "tabllm":
        train_lines = []
        for feat, lbl in zip(X_train_arr, y_train_arr):
            feature_pairs = [f"{feature_names[i]} is {feat[i]}" for i in range(len(feature_names))]
            line = ", ".join(feature_pairs) + f", Label is {lbl}"
            train_lines.append(line)

        test_lines = []
        for i, feat in enumerate(X_test_arr):
            feature_pairs = [f"{feature_names[j]} is {feat[j]}" for j in range(len(feature_names))]
            line = f"ID {i}: " + ", ".join(feature_pairs)
            test_lines.append(line)
    else:
        # concat style
        train_lines = [",".join(map(str, feat)) + "|" + str(lbl) for feat, lbl in zip(X_train_arr, y_train_arr)]
        test_lines = [f"ID {i}:" + ",".join(map(str, feat)) for i, feat in enumerate(X_test_arr)]

    return train_lines, test_lines


def create_labels_json(y_test) -> List[dict]:
    """Create assistant payload JSON array describing labels (used as assistant content)."""
    return [{"id": i, "label": int(lbl)} for i, lbl in enumerate(y_test)]


def build_user_prompt(
    feature_num: int,
    label_set: List[int],
    feature_names: List[str],
    train_lines: List[str],
    test_lines: List[str],
    prompt_format_style: str = "concat",
    include_feature_descriptions: bool = False,
) -> str:
    """
    Build the user-facing prompt string (the text the model will see) based
    on the requested formatting style.
    """
    if prompt_format_style == "tabllm":
        user_prompt = f"""
[Data]
• Each sample contains {feature_num} features and 1 label. Label set = {label_set}.
• Format: FeatureName is value, FeatureName is value, ..., Label is label_value
• Features: {', '.join(feature_names)}

[Training set]  (order of rows does NOT matter)  
{len(train_lines)} rows:  
{chr(10).join(train_lines)}

[Test set]  (keep original order!)  
Each row = ID, then feature assignments (ID is NOT a feature).  
{len(test_lines)} rows:  
{chr(10).join(test_lines)}

[Output requirements]
Return **only** a JSON array.
Each element: {{"id": <ID>, "label": <predicted_label>}}

Begin when ready. Do not output anything except the JSON array.
"""
    else:
        feature_description_line = ""
        if include_feature_descriptions:
            feature_description_line = f"• Feature descriptions  (in order) \n {','.join([f'{name}' for i, name in enumerate(feature_names)])}\n"

        user_prompt = f"""
[Data]
• Each sample = {feature_num} features + 1 label.  Label set = {label_set}.  
• Features in a row are comma-separated.  Features and label are separated by "|".
{feature_description_line}

[Training set]  (order of rows does NOT matter)  
{len(train_lines)} rows:  
{chr(10).join(train_lines)}

[Test set]  (keep original order!)  
Each row = ID, then N features (ID is NOT a feature).  
{len(test_lines)} rows:  
{chr(10).join(test_lines)}

[Output requirements]
Return **only** a JSON array.
Each element: {{"id": <ID>, "label": <predicted_label>}}

Begin when ready. Do not output anything except the JSON array.
"""
    return user_prompt


def build_full_prompt(system_prompt: str, user_prompt: str, labels_json: List[dict]) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(labels_json, ensure_ascii=False)}
        ]
    }


def ensure_1d(y):
    """Ensure label-like object is returned as a 1-D numpy array.

    If `y` is a scalar, convert to a 1-length numpy array; otherwise return as-is.
    """
    if np.isscalar(y):
        return np.array([y])
    return y
