from pathlib import Path
import json
import random
from math import comb

import numpy as np
import pandas as pd
import tiktoken
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from tabpfn import TabPFNClassifier

# for GPT post-training, other LLMs need adjustment
enc = tiktoken.get_encoding("cl100k_base")

# Model configuration
model_name = "rf"
rf_model = RandomForestClassifier(n_estimators=30, random_state=42)
pfn = TabPFNClassifier()

TOKEN_LIMIT = 110000
num_test = 20

output_file = f"ticl_500k_10_1024_{num_test}_shuffle_token_0812_real_resample.jsonl"
split_data_dir = "ticl_scmcsvfile_10_1024_50_shuffle_0711"

X_test_dir = Path(split_data_dir) / "X_test"
X_train_dir = Path(split_data_dir) / "X_train"
y_test_dir = Path(split_data_dir) / "y_test"
y_train_dir = Path(split_data_dir) / "y_train"


def _binom_sf(k: int, n: int, p: float) -> float:
    """Right tail: P[X >= k] for X ~ Bin(n, p)."""
    prob = 0.0
    for i in range(k, n + 1):
        prob += comb(n, i) * (p**i) * ((1 - p) ** (n - i))
    return prob


def better_than_random_guard(
    y_true,
    y_pred,
    *,
    # Hyperparameters (tuned for n ≈ 20)
    alpha=0.2,  # one-sided binomial test significance level
    delta_bacc=0.03,  # balanced accuracy must exceed 1/C by at least this margin
    max_dom_frac=0.85,  # upper bound on dominant predicted class fraction
    min_pred_classes=2,  # minimum number of distinct predicted classes
    require_two_nonzero_f1=True,  # require at least two classes with F1 > 0
    min_n=20,  # minimum number of test samples
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same length"
    n = len(y_true)

    # Sample size and number of classes
    classes, counts = np.unique(y_true, return_counts=True)
    C = len(classes)
    if n < min_n or C < 2:
        # Too few samples or only a single true class: we do not treat this as better than random
        return {
            "n": n,
            "C": C,
            "pass_flag": False,
            "note": "too_few_samples_or_single_class",
        }

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=classes)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_pc = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Strong random baseline p0: max(prior random guessing, majority class)
    props = counts / n
    p0_prior = float(np.sum(props**2))
    p0_major = float(props.max())
    p0 = max(p0_prior, p0_major)

    # One-sided binomial test: is acc significantly greater than p0?
    k = int(round(acc * n))
    pval = _binom_sf(k, n, p0)

    # Macro-F1 of a classifier that always predicts the majority class
    maj_class = classes[np.argmax(counts)]
    f1_macro_maj = f1_score(
        y_true, np.full(n, maj_class), average="macro", zero_division=0
    )

    # Collapse checks
    uniq_pred = len(np.unique(y_pred))
    _, pred_idx = np.unique(y_pred, return_inverse=True)
    dom_frac = (np.bincount(pred_idx).max() / n) if n > 0 else 1.0
    nonzero_f1 = int(np.sum(f1_pc > 0))

    # Individual conditions
    _cond1 = pval < alpha
    _cond2 = kappa > 0.0
    _cond3 = bacc > (1.0 / C + delta_bacc)
    _cond4 = f1_macro >= f1_macro_maj
    _cond5 = uniq_pred >= min_pred_classes
    _cond6 = dom_frac <= max_dom_frac
    _cond7 = nonzero_f1 >= 2 if require_two_nonzero_f1 else True

    pass_flag = all([_cond1, _cond2, _cond3, _cond4, _cond5, _cond6, _cond7])

    if not pass_flag:
        reasons = []
        if not _cond1:
            reasons.append(
                f"[statistical significance] pval({pval:.4g}) >= alpha({alpha:.3g}); baseline p0={p0:.3f}"
            )
        if not _cond2:
            reasons.append(f"[chance correction] kappa({kappa:.3f}) <= 0")
        if not _cond3:
            reasons.append(
                f"[balanced accuracy] BA({bacc:.3f}) <= 1/C+δ({1.0 / C + delta_bacc:.3f}) [C={C}]"
            )
        if not _cond4:
            reasons.append(
                f"[macro-F1 baseline] macro-F1({f1_macro:.3f}) < macro-F1(majority-only)({f1_macro_maj:.3f})"
            )
        if not _cond5:
            reasons.append(
                f"[collapse - #classes] number of predicted classes uniq_pred({uniq_pred}) < required({min_pred_classes})"
            )
        if not _cond6:
            reasons.append(
                f"[collapse - dominant fraction] dominant class fraction dom_frac({dom_frac:.3f}) > max_dom_frac({max_dom_frac:.2f})"
            )
        if not _cond7:
            reasons.append(
                f"[collapse - effective F1] number of classes with non-zero F1({nonzero_f1}) < 2"
            )

        print("Failed better_than_random_guard:")
        for r in reasons:
            print(" -", r)

    return {
        "n": n,
        "C": C,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "kappa": kappa,
        "f1_macro": f1_macro,
        "f1_per_class": f1_pc.tolist(),
        "p0": p0,
        "pval_vs_p0": pval,
        "f1_macro_majority": f1_macro_maj,
        "uniq_pred": uniq_pred,
        "dom_frac": dom_frac,
        "nonzero_f1": nonzero_f1,
        "pass_flag": pass_flag,
    }


f = open(output_file, "w", encoding="utf-8")
cnt = 0
cnt_process = 0

for X_test_file in X_test_dir.glob("*.csv"):
    fname = X_test_file.name
    X_train_file = X_train_dir / fname
    y_test_file = y_test_dir / fname
    y_train_file = y_train_dir / fname
    print(X_test_file, X_train_file, y_test_file, y_train_file)

    # Optional: ensure all files exist for this split
    if not (X_train_file.exists() and y_test_file.exists() and y_train_file.exists()):
        print(f"⚠️ Missing files for: {fname}")
        continue

    X_test = pd.read_csv(X_test_file)
    X_train = pd.read_csv(X_train_file)
    y_test = pd.read_csv(y_test_file).squeeze()   # convert to Series
    y_train = pd.read_csv(y_train_file).squeeze() # convert to Series

    # Token-limit-aware cap based on feature dimension
    feature_num = max(1, X_train.shape[1])
    cap = 7500 // feature_num
    num_train = min(len(X_train), cap)
    num_train = int(num_train * random.uniform(0.7, 1.0))
    num_train = max(num_train, 1)

    # Shuffle and subsample (keep X/y aligned)
    perm = np.random.permutation(len(X_train))
    keep = perm[:num_train]
    X_train = X_train.iloc[keep].reset_index(drop=True)
    y_train = y_train.iloc[keep].reset_index(drop=True)

    # Use first num_test samples from the test split
    X_test = X_test.iloc[:num_test].reset_index(drop=True)
    y_test = y_test.iloc[:num_test].reset_index(drop=True)

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

    # y_train may be a DataFrame or Series; normalize to a 1D Series for distribution stats
    y_train_series = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train

    # Class counts and proportions (dropna=False keeps any NaN label as its own category)
    counts = y_train_series.value_counts(dropna=False)
    props = y_train_series.value_counts(normalize=True, dropna=False)

    dist_df = (
        pd.DataFrame({"count": counts, "ratio": props})
        .sort_index()
    )
    print("\n[Train label distribution]")
    print(
        dist_df.to_string(
            formatters={"count": "{:d}".format, "ratio": "{:.3f}".format}
        )
    )

    # Extra: number of classes and majority-class statistics
    C = len(dist_df)
    maj_cls = dist_df["ratio"].idxmax()
    maj_frac = dist_df["ratio"].max()
    print(f"classes={C}, majority_class={maj_cls}, majority_ratio={maj_frac:.3f}")

    # Sanity check: same type for predictions and ground truth
    assert type(pred_test) == type(y_test), "pred_test and y_test have different types"

    print(
        "num_train",
        num_train,
        "feature_num",
        feature_num,
        "accuracy",
        accuracy,
        "f1_per_class",
        f1_per_class,
    )

    m = better_than_random_guard(
        y_test, pred_test, alpha=0.2, delta_bacc=0.03, max_dom_frac=0.95
    )
    cnt += 1
    if not m["pass_flag"]:
        print("*****pass******", cnt, cnt_process)
        continue

    cnt_process += 1
    print("-----process-----", cnt, cnt_process)

    # Rescale feature values for token-friendly integer representation
    try:
        X_train = ((X_train * 120 + 500).clip(lower=0)).astype(int)
        X_test = ((X_test * 120 + 500).clip(lower=0)).astype(int)
    except Exception:
        # e.g., inf or NA issues
        continue

    # Convert training set to "feature1,feature2,...|label" lines
    label_set = []
    train_lines = []
    for features, label in zip(X_train.values, y_train):
        feature_str = ",".join(map(str, features))
        train_lines.append(f"{feature_str}|{label}")
        if label not in label_set:
            label_set.append(label)
    label_set.sort()

    # Convert test set to "id:feature1,feature2,..."
    test_lines = []
    labels_json = []
    for idx, (features, label) in enumerate(zip(X_test.values, y_test), 0):
        feature_str = ",".join(map(str, features))
        test_lines.append(f"{idx}:{feature_str}")
        labels_json.append(
            {
                "id": idx,
                "label": label,  # ground-truth label
            }
        )

    test_lines = test_lines[:num_test]
    labels_json = labels_json[:num_test]
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
    if num_tokens > TOKEN_LIMIT:
        continue

    json.dump(prompt, f, ensure_ascii=False)
    f.write("\n")
    f.flush()

f.close()
