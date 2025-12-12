import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from tabpfn import TabPFNClassifier

# for GPT post-training, other LLMs need adjustment
enc = tiktoken.get_encoding("cl100k_base")

# Model configuration
model_name = "rf"
rf_model = RandomForestClassifier(n_estimators=30, random_state=42)
pfn = TabPFNClassifier()

TOKEN_LIMIT = 110000

output_file = "ticl_500k_10_1024_50_random_dual_distribution_shuffle_token_0712_rf.jsonl"
split_data_dir = "ticl_scmcsvfile_10_1024_50_shuffle_0711"

X_test_dir = Path(split_data_dir) / "X_test"
X_train_dir = Path(split_data_dir) / "X_train"
y_test_dir = Path(split_data_dir) / "y_test"
y_train_dir = Path(split_data_dir) / "y_train"

f = open(output_file, "w", encoding="utf-8")

for X_test_file in X_test_dir.glob("*.csv"):
    fname = X_test_file.name
    X_train_file = X_train_dir / fname
    y_test_file = y_test_dir / fname
    y_train_file = y_train_dir / fname
    print(X_test_file, X_train_file, y_test_file, y_train_file)

    # Optional: ensure all files for this split exist
    if not (X_train_file.exists() and y_test_file.exists() and y_train_file.exists()):
        print(f"⚠️ Missing files for: {fname}")
        continue

    X_test = pd.read_csv(X_test_file)
    X_train = pd.read_csv(X_train_file)
    y_test = pd.read_csv(y_test_file).squeeze()   # convert to Series
    y_train = pd.read_csv(y_train_file).squeeze()

    if model_name == "rf":
        try:
            rf_model.fit(X_train, y_train)
            pred_test = rf_model.predict(X_test)
        except Exception:
            continue
    elif model_name == "pfn":
        try:
            pfn.fit(X_train, y_train)
            pred_test = pfn.predict(X_test)
        except Exception:
            continue
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    accuracy = accuracy_score(y_test, pred_test)
    f1_macro = f1_score(y_test, pred_test, average="macro")
    f1_per_class = f1_score(y_test, pred_test, average=None)

    pred_test = pd.Series(pred_test, index=y_test.index, name=y_test.name)

    # Ensure types are consistent
    assert type(pred_test) == type(y_test), "pred_test and y_test have different types"

    correct_indices = pred_test[pred_test == y_test].index

    print("accuracy", accuracy, "f1", f1_per_class)

    # Filter out low-quality splits
    if accuracy < 0.5 or len(f1_per_class) < 2 or sorted(f1_per_class)[-2] < 0.25:
        print("pass")
        continue

    # Compute label distribution in the training set
    train_counter = Counter(y_train)
    total_train = sum(train_counter.values())
    train_ratio = {lbl: cnt / total_train for lbl, cnt in train_counter.items()}

    # Collect indices of correctly predicted samples for each label
    correct_by_label = {}
    for lbl in train_ratio.keys():
        idx_lbl = correct_indices[y_test.loc[correct_indices] == lbl]
        if len(idx_lbl):
            correct_by_label[lbl] = idx_lbl

    # Allocate a quota of 20 examples according to training label distribution
    quota = {
        lbl: max(1, int(round(r * 20)))  # at least 1 per label
        for lbl, r in train_ratio.items()
    }

    # Adjust quota so that the total number is exactly 20
    while sum(quota.values()) != 20:
        diff = 20 - sum(quota.values())
        # assign the difference to the label that has most samples in training
        target = max(train_counter, key=train_counter.get)
        quota[target] += diff

    # Sample indices according to the quota
    selected_indices = []

    for lbl, n in quota.items():
        pool = correct_by_label.get(lbl, [])
        if len(pool) >= n:
            # enough samples → sample without replacement
            sel = np.random.choice(pool, size=n, replace=False)
        elif len(pool) > 0:
            # not enough samples → sample with replacement
            sel = np.random.choice(pool, size=n, replace=True)
        else:
            # no correctly predicted sample for this label; will be filled later
            continue
        selected_indices.extend(sel)

    # If still fewer than 20, fill from any available label pools
    while len(selected_indices) < 20:
        for lbl, pool in correct_by_label.items():
            if len(selected_indices) >= 20:
                break
            sel = np.random.choice(pool, size=1)
            selected_indices.append(sel[0])

    selected_indices = pd.Index(selected_indices)
    selected_indices = pd.Index(np.random.permutation(selected_indices))

    # Rescale features into integer range for token-friendly representation
    try:
        X_train = ((X_train * 120 + 500).clip(lower=0)).astype(int)
        X_test = ((X_test * 120 + 500).clip(lower=0)).astype(int)
    except Exception:
        # e.g., inf or NA in features
        continue

    # Keep only the selected 20 samples in test / predictions / labels
    X_test_selected = X_test.loc[selected_indices]
    pred_test_selected = pred_test.loc[selected_indices]
    y_test_selected = y_test.loc[selected_indices]

    label_set = []
    train_lines = []

    # Convert training set to "feature1,feature2,...|label"
    for features, label in zip(X_train.values, y_train):
        feature_str = ",".join(map(str, features))
        train_lines.append(f"{feature_str}|{label}")
        if label not in label_set:
            label_set.append(label)
    label_set.sort()

    # Convert test set to "id:feature1,feature2,..."
    test_lines = []
    labels_json = []
    for idx, (features, label) in enumerate(
        zip(X_test_selected.values, y_test_selected), 0
    ):
        feature_num = features.shape[0]
        feature_str = ",".join(map(str, features))
        test_lines.append(f"{idx}:{feature_str}")
        labels_json.append(
            {
                "id": idx,
                "label": label,  # ground-truth label
            }
        )

    train_str = "\n".join(train_lines)
    test_str = "\n".join(test_lines)
    test_labels = labels_json

    system_prompt = "You are an AI assistant. Your task is supervised classification."

    user_prompt = f"""
    [Data]
    • Each sample = {feature_num} features + 1 label.  Label set = {label_set}.  
    • Features in a row are comma-separated.  Features and label are separated by “|”.

    [Training set]  (order of rows does NOT matter)  
    {len(train_lines)} rows:  
    {train_str}

    [Test set]  (keep original order!)  
    Each row = ID, then {feature_num} features (ID is NOT a feature).  
    {len(test_lines)} rows:  
    {test_str}

    [Output requirements]
    Return **only** a JSON array.  
    Each element: {{\"id\": <ID>, \"label\": <predicted_label>}}.

    Example (for two test cases):
    [
      {{\"id\": \"0\", \"label\": 1}},
      {{\"id\": \"1\", \"label\": 0}}
    ]

    Begin when ready. Do not output anything except the JSON array.
    """

    assistant_prompt = json.dumps(test_labels, ensure_ascii=False)

    prompt = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt},
        ]
    }
    num_tokens = len(enc.encode(str(prompt)))
    print(len(train_lines), feature_num, num_tokens, len(str(prompt)))
    if num_tokens > TOKEN_LIMIT:
        continue

    json.dump(prompt, f, ensure_ascii=False)
    f.write("\n")
    f.flush()

f.close()
