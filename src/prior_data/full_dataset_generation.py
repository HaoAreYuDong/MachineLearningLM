import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
from pathlib import Path
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
import time
#time.sleep(60*60)
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, balanced_accuracy_score, cohen_kappa_score, f1_score
)
from collections import Counter
import random
from math import comb
#import time
#time.sleep(60*60*4)
# 初始化随机森林模型
model_name = "rf"
rf_model = RandomForestClassifier(n_estimators=30, random_state=42)
pfn = TabPFNClassifier()

TOEKN_LIMIT = 110000

num_test = 20
output_file = f'ticl_500k_10_1024_{num_test}_shuffle_token_0812_real_resample.jsonl'
split_data_dir = "ticl_scmcsvfile_10_1024_50_shuffle_0711" 

X_test_dir = Path(split_data_dir + '/X_test')
X_train_dir = Path(split_data_dir + '/X_train')
y_test_dir = Path(split_data_dir + '/y_test')
y_train_dir = Path(split_data_dir + '/y_train')

def _binom_sf(k: int, n: int, p: float) -> float:
    """右尾：P[X >= k],  X ~ Bin(n, p)"""
    prob = 0.0
    for i in range(k, n + 1):
        prob += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return prob

def better_than_random_guard(
    y_true,
    y_pred,
    *,
    # —— 超参数（为 n=20 调的默认值）——
    alpha=0.2,           # 单侧二项检验显著性阈：n=20 时略放宽到 0.10，避免低功效
    delta_bacc=0.03,      # Balanced Accuracy 必须超出随机基线 1/C 的幅度
    max_dom_frac=0.85,    # 预测主导类别的占比上限（防止几乎全预测成一个类）
    min_pred_classes=2,   # 至少预测出这么多个不同类别
    require_two_nonzero_f1=True,  # 至少两个类别的 F1 > 0（防塌缩）
    min_n=20              # 最小测试样本数
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "y_true / y_pred 长度不一致"
    n = len(y_true)

    # 样本数与类别数
    classes, counts = np.unique(y_true, return_counts=True)
    C = len(classes)
    if n < min_n or C < 2:
        # 测试集太小 / 只有一个真类时，不判优于随机
        return {
            "n": n, "C": C, "pass_flag": False, "note": "too_few_samples_or_single_class"
        }

    # —— 基本指标 —— 
    acc      = accuracy_score(y_true, y_pred)
    bacc     = balanced_accuracy_score(y_true, y_pred)
    kappa    = cohen_kappa_score(y_true, y_pred, labels=classes)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_pc    = f1_score(y_true, y_pred, average=None, zero_division=0)

    # —— 强随机基线 p0 ——（择严）
    # 按真实先验随机抽样的期望准确率（prior）与“永远猜多数类”（majority）取较大者
    props    = counts / n
    p0_prior = float(np.sum(props ** 2))
    p0_major = float(props.max())
    p0       = max(p0_prior, p0_major)

    # —— 单侧二项检验：acc 是否显著高于 p0 —— 
    k     = int(round(acc * n))
    pval  = _binom_sf(k, n, p0)

    # —— 与“全预测多数类”的宏 F1 比较 —— 
    maj_class     = classes[np.argmax(counts)]
    f1_macro_maj  = f1_score(y_true, np.full(n, maj_class), average="macro", zero_division=0)

    # —— 反塌缩检查 —— 
    uniq_pred = len(np.unique(y_pred))
    _, pred_idx = np.unique(y_pred, return_inverse=True)
    dom_frac  = (np.bincount(pred_idx).max() / n) if n > 0 else 1.0
    nonzero_f1 = int(np.sum(f1_pc > 0))

    # 逐项条件布尔值
    _cond1 = (pval < alpha)
    _cond2 = (kappa > 0.0)  # 或 > 0.05 更严格
    _cond3 = (bacc > (1.0 / C + delta_bacc))
    _cond4 = (f1_macro >= f1_macro_maj)
    _cond5 = (uniq_pred >= min_pred_classes)
    _cond6 = (dom_frac <= max_dom_frac)
    _cond7 = (nonzero_f1 >= 2) if require_two_nonzero_f1 else True

    pass_flag = all([_cond1, _cond2, _cond3, _cond4, _cond5, _cond6, _cond7])

    if not pass_flag:
        reasons = []
        if not _cond1:
            reasons.append(f"[统计显著性] pval({pval:.4g}) >= alpha({alpha:.3g}); 基线p0={p0:.3f}")
        if not _cond2:
            reasons.append(f"[机会校正] kappa({kappa:.3f}) <= 0")
        if not _cond3:
            reasons.append(f"[均衡准确率] BA({bacc:.3f}) <= 1/C+δ({1.0/C + delta_bacc:.3f}) [C={C}]")
        if not _cond4:
            reasons.append(f"[宏F1基线] macro-F1({f1_macro:.3f}) < macro-F1(全多数类)({f1_macro_maj:.3f})")
        if not _cond5:
            reasons.append(f"[塌缩-类别数] 预测类别数 uniq_pred({uniq_pred}) < 最小要求({min_pred_classes})")
        if not _cond6:
            reasons.append(f"[塌缩-主导比例] 主导类别占比 dom_frac({dom_frac:.3f}) > 上限({max_dom_frac:.2f})")
        if not _cond7:
            reasons.append(f"[塌缩-有效F1] 非零F1的类别数({nonzero_f1}) < 2")

        print("未通过判定：")
        for r in reasons:
            print(" -", r)

    return {
        "n": n, "C": C,
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
        "pass_flag": pass_flag
    }
    
f = open(output_file, 'w', encoding='utf-8')
cnt = 0 
cnt_process = 0
for X_test_file in X_test_dir.glob("*.csv"):
    fname = X_test_file.name          # 带扩展名 (如 "foo.csv")
    X_train_file = X_train_dir / fname
    y_test_file  = y_test_dir  / fname
    y_train_file = y_train_dir / fname
    print(X_test_file,X_train_file,y_test_file,y_train_file)
    # 可选：确认文件都真的存在
    if not (X_train_file.exists() and y_test_file.exists() and y_train_file.exists()):
        print(f"⚠️ 缺少文件：{fname}")
        continue
    X_test = pd.read_csv(X_test_file)
    X_train = pd.read_csv(X_train_file)
    y_test = pd.read_csv(y_test_file).squeeze() # 转换为Series
    y_train = pd.read_csv(y_train_file).squeeze()
    
    feature_num = max(1, X_train.shape[1])  # 防止除零
    # token limit restriction of qwen's tokenizer
    cap = 7500 // feature_num
    num_train = min(len(X_train), cap)
    # randomly sample a subset
    num_train = int(num_train * random.uniform(0.7, 1.0))
    num_train = max(num_train, 1)

    # 打乱并同步抽样（保持 X/y 对齐）
    perm = np.random.permutation(len(X_train))
    keep = perm[:num_train]
    X_train = X_train.iloc[keep].reset_index(drop=True)
    y_train = y_train.iloc[keep].reset_index(drop=True)
    
    X_test = X_test.iloc[:num_test].reset_index(drop=True)
    y_test = y_test.iloc[:num_test].reset_index(drop=True)

    if model_name == "rf":
        try:
            # 拟合模型
            rf_model.fit(X_train, y_train)
            pred_test = rf_model.predict(X_test)
        except:
            continue
    if model_name == "pfn":
        try:
            # 拟合模型
            pfn.fit(X_train, y_train)
            pred_test = pfn.predict(X_test)
        except:
            continue

    accuracy = accuracy_score(y_test, pred_test)
    f1_macro = f1_score(y_test, pred_test, average='macro')
    f1_per_class = f1_score(y_test, pred_test, average=None)
    
    pred_test = pd.Series(pred_test, index=y_test.index, name=y_test.name)

    y_train_series = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train

    # 各类别计数与占比（dropna=False 保留缺失值一并统计）
    counts = y_train_series.value_counts(dropna=False)
    props  = y_train_series.value_counts(normalize=True, dropna=False)

    dist_df = pd.DataFrame({"count": counts, "ratio": props}).sort_index()
    print("\n[Train y distribution]")
    print(dist_df.to_string(formatters={"count": "{:d}".format, "ratio": "{:.3f}".format}))

    # 额外：类别数、最多类别及占比
    C = len(dist_df)
    maj_cls = dist_df["ratio"].idxmax()
    maj_frac = dist_df["ratio"].max()
    print(f"classes={C}, majority_class={maj_cls}, majority_ratio={maj_frac:.3f}")


    # 验证数据类型是否一致
    assert type(pred_test) == type(y_test), "pred_test 与 y_test 类型不一致"
    print("num_train", num_train, "feature_num", feature_num, "accuracy", accuracy, "f1_per_class", f1_per_class)
    m = better_than_random_guard(y_test, pred_test, alpha=0.2, delta_bacc=0.03, max_dom_frac=0.95)
    cnt += 1
    if not m["pass_flag"]:
        print("*****pass******", cnt, cnt_process)
        continue
    cnt_process += 1
    print("-----process-----", cnt, cnt_process)
    # 打印最大最小值（按列）
    try:
        X_train = ((X_train * 120 + 500).clip(lower=0)).astype(int)
        X_test = ((X_test * 120 + 500).clip(lower=0)).astype(int)
    except:
        # maybe inf or NA
        continue
    # 打印最大最小值（按列）
    #print("\nX_train 每列最大值：")
    #print(X_train.max())
    label_set = []
    # 转换训练集格式
    train_lines = []
    for features, label in zip(X_train.values, y_train):
        feature_str = ",".join(map(str, features))
        train_lines.append(f"{feature_str}|{label}")
        if label not in label_set:
            label_set.append(label)
    label_set.sort()
    
    # 转换测试集格式
    test_lines = []
    labels_json = []
    for idx, (features, label) in enumerate(zip(X_test.values, y_test), 0):
        # 生成唯一ID（格式：id_001）
        feature_num = features.shape[0]
        feature_str = ",".join(map(str, features))
        #input(feature_str)
        test_lines.append(f"{idx}:{feature_str}")
        labels_json.append({
        "id": idx,
        "label": label  # 实际标签
    })
    
    
    test_lines = test_lines[:num_test]
    labels_json = labels_json[:num_test]
    train_str = "\n".join(train_lines)
    test_str = "\n".join(test_lines) 
    test_labels = labels_json
    #print(train_str)
    #print(test_str)
    
    system_prompt = '''You are an AI assistant. Your task is supervised classification.'''

    user_prompt = f'''
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
    '''

    assistant_prompt = json.dumps(test_labels, ensure_ascii=False)

    prompt = {"messages": [{"role": "system", "content":system_prompt }, {"role": "user", "content":user_prompt }, {"role": "assistant", "content":assistant_prompt }]}
    num_tokens = len(enc.encode(str(prompt)))
    if num_tokens > TOEKN_LIMIT :
        #input()
        continue
    json.dump(prompt, f, ensure_ascii=False)  # 写入单个对象
    f.write("\n")  # 添加换行符分隔
    f.flush()
f.close()