# mmlu_eval_multi.py
# Evaluate multiple HuggingFace causal LMs on MMLU with k-shot prompting.
# - Models: TabuLa-8B, Qwen2.5-7B-Instruct, MachineLearningLM-7B-v1
# - Metrics: micro accuracy (overall), macro accuracy (avg over subjects)
# - Implementation: scores conditional log-likelihood of " Answer: X" for X in {A,B,C,D}
# - Notes:
#   * Uses validation split as the few-shot pool (by subject), test split for evaluation.
#   * Truncates long prompts to model context length (best-effort).
#   * Uses bfloat16 by default where supported.

import os
import math
import random
import collections
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# -------------------------
# Configuration
# -------------------------
DEVICE_ID = int(os.environ.get("CUDA_ID", "3"))    # set CUDA device via env var, e.g., CUDA_ID=3
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16  # change to torch.float16 for fp16 GPUs; or None for fp32

# Models to evaluate (name, trust_remote_code)
MODELS = [
    ("MachineLearningLM/MachineLearningLM-7B-v1", True), # MachineLearningLM 7B v1
    ("mlfoundations/tabula-8b", False),                  # TabuLa-8B
    ("Qwen/Qwen2.5-7B-Instruct", True),                  # Qwen 2.5 7B Instruct
]

K_SHOTS = [0, 5]  # k-shot settings to sweep
BATCH_SIZE_CONTEXT = 4        # number of contexts per forward pass when scoring a single letter
PRINT_PER_SUBJECT = True      # set False if you only want aggregated metrics
SEED = 42                     # for per-subject shuffling of the few-shot pool
CHOICE_LETTERS = ["A", "B", "C", "D"]

# -------------------------
# Data loading (MMLU)
# -------------------------
print("Using device:", DEVICE)
print("Loading MMLU...")
ds = load_dataset("cais/mmlu", "all")
val_ds = ds["validation"]  # few-shot pool
test_ds = ds["test"]       # evaluation

# Group validation examples by subject for same-subject few-shot selection
val_by_subject: Dict[str, List[dict]] = collections.defaultdict(list)
for ex in val_ds:
    val_by_subject[ex["subject"]].append(ex)

rng = random.Random(SEED)
for sub in val_by_subject:
    rng.shuffle(val_by_subject[sub])

# -------------------------
# Prompt formatting
# -------------------------
def format_example(ex: dict, include_answer: bool) -> str:
    """Serialize one MMLU example as text. If include_answer=True, append 'Answer: X'."""
    q, choices, ans_idx = ex["question"], ex["choices"], ex["answer"]
    lines = [q] + [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
    if include_answer:
        lines += [f"Answer: {CHOICE_LETTERS[ans_idx]}"]
    else:
        lines += ["Answer:"]
    return "\n".join(lines)

def build_kshot_prompt(test_ex: dict, k: int) -> str:
    """Use k examples from the same subject (from validation) as demonstrations, then the test question without the answer."""
    sub = test_ex["subject"]
    pool = val_by_subject[sub]
    k_eff = min(k, len(pool))
    demos = [format_example(pool[i], include_answer=True) for i in range(k_eff)]
    tgt = format_example(test_ex, include_answer=False)
    return "\n\n".join(demos + [tgt])

# -------------------------
# Scoring utilities
# -------------------------
@torch.no_grad()
def score_batch_letter(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    context_ids_list: List[List[int]],
    cont_text: str,
) -> List[float]:
    """
    For a batch of contexts, append cont_text (e.g., ' A') and compute conditional log-likelihood.
    Returns a list of summed log-probs (higher is better).
    """
    # Encode the continuation once
    cont_ids = tok.encode(cont_text, add_special_tokens=False)

    # Build input and label tensors
    batch_inputs, batch_labels = [], []
    for ctx in context_ids_list:
        ids = ctx + cont_ids
        labels = [-100] * len(ctx) + cont_ids
        batch_inputs.append(torch.tensor(ids, dtype=torch.long))
        batch_labels.append(torch.tensor(labels, dtype=torch.long))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        batch_inputs, batch_first=True, padding_value=tok.pad_token_id
    ).to(device)
    labels = torch.nn.utils.rnn.pad_sequence(
        batch_labels, batch_first=True, padding_value=-100
    ).to(device)
    attn = (input_ids != tok.pad_token_id).to(device)

    # Forward
    if DTYPE is not None:
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE, enabled=device.type == "cuda"):
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    else:
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)

    logits = out.logits  # [B, T, V]

    # Sum token log-probs on the continuation span
    logprobs = []
    for b in range(input_ids.size(0)):
        # first index in labels with target tokens
        start = (labels[b] != -100).nonzero(as_tuple=False)[0].item()
        lp = 0.0
        # at time t, the target token is input_ids[b, start + t]
        # we use logits from the previous position (start + t - 1)
        for t, tgt in enumerate(input_ids[b, start:start + len(cont_ids)]):
            pos = max(start + t - 1, 0)
            lp += torch.log_softmax(logits[b, pos], dim=-1)[tgt].item()
        logprobs.append(lp)
    return logprobs

def maybe_truncate_to_ctx(tok, model, ids: List[int], extra_len: int) -> List[int]:
    """
    Truncate the left side of 'ids' to fit model's max length, allowing room for 'extra_len' continuation tokens and a small margin.
    """
    # Best-effort max length detection
    max_len = getattr(model.config, "max_position_embeddings", None)
    if max_len is None:
        # Reasonable default for many 7B/8B models
        max_len = 4096

    margin = 8
    allow = max_len - extra_len - margin
    if allow <= 0:
        # Fallback: keep at least some prompt (this is pathological)
        allow = max(16, max_len // 2)
    if len(ids) > allow:
        return ids[-allow:]  # keep the last tokens (most recent instructions)
    return ids

@torch.no_grad()
def evaluate_model_on_mmlu(
    model_name: str,
    trust_remote_code: bool,
    k_shot: int,
    batch_size_context: int = 4,
    print_per_subject: bool = True,
) -> Tuple[float, float]:
    """
    Evaluate one model at a specific k-shot on MMLU test set.
    Returns (micro_accuracy, macro_accuracy).
    """
    print(f"\nLoading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE if (DTYPE is not None and DEVICE.type == "cuda") else None,
        trust_remote_code=trust_remote_code,
    ).to(DEVICE).eval()

    if tok.pad_token_id is None:
        # Ensure a pad token exists (many causal LMs reuse EOS)
        tok.pad_token = tok.eos_token

    # Pre-encode all prompts (with truncation if needed)
    print(f"Building {k_shot}-shot prompts...")
    raw_prompts = [build_kshot_prompt(ex, k=k_shot) for ex in test_ds]
    # For each prompt, we will later append " A" | " B" | " C" | " D"
    ctx_ids = []
    # Pre-encode once to get continuation len (identical across prompts per letter)
    cont_ids_A = tok.encode(" " + CHOICE_LETTERS[0], add_special_tokens=False)
    for p in raw_prompts:
        ids = tok.encode(p, add_special_tokens=False)
        ids = maybe_truncate_to_ctx(tok, model, ids, extra_len=len(cont_ids_A))
        ctx_ids.append(ids)

    answers = [ex["answer"] for ex in test_ds]
    subjects = [ex["subject"] for ex in test_ds]
    preds = [None] * len(test_ds)

    # Score each batch of contexts for all 4 options
    print("Scoring...")
    total_batches = math.ceil(len(ctx_ids) / batch_size_context)
    pbar = tqdm(total=total_batches * len(CHOICE_LETTERS),
                desc=f"[{model_name}] k={k_shot} scoring",
                leave=False)

    for start in range(0, len(ctx_ids), batch_size_context):
        end = min(len(ctx_ids), start + batch_size_context)
        batch_ctx = ctx_ids[start:end]

        scores_per_choice = []
        for letter in CHOICE_LETTERS:
            cont_text = " " + letter
            scores = score_batch_letter(tok, model, DEVICE, batch_ctx, cont_text)
            scores_per_choice.append(scores)
            pbar.update(1)  # one letter scored for this batch

        # Pick argmax per example
        for i in range(end - start):
            s = [scores_per_choice[j][i] for j in range(len(CHOICE_LETTERS))]
            preds[start + i] = int(torch.tensor(s).argmax().item())

    pbar.close()


    # Compute micro accuracy
    correct = sum(int(p == a) for p, a in zip(preds, answers))
    micro = correct / len(test_ds)

    # Compute macro accuracy by subject
    by_subj = collections.defaultdict(lambda: {"c": 0, "n": 0})
    for p, a, sub in zip(preds, answers, subjects):
        by_subj[sub]["n"] += 1
        by_subj[sub]["c"] += int(p == a)
    macro = sum(v["c"] / v["n"] for v in by_subj.values()) / len(by_subj)

    print(f"\n[{model_name}] [k-shot={k_shot}] Micro accuracy: {micro:.4f}")
    print(f"[{model_name}] [k-shot={k_shot}] Macro accuracy (avg over {len(by_subj)} subjects): {macro:.4f}")

    if print_per_subject:
        print("\nPer-subject accuracy:")
        for sub in sorted(by_subj.keys()):
            n = by_subj[sub]["n"]
            acc = by_subj[sub]["c"] / n
            print(f"{sub:35s}  n={n:4d}  acc={acc:.4f}")

    # Cleanup to reduce VRAM/RAM before next model
    del model
    torch.cuda.empty_cache()
    return micro, macro

# -------------------------
# Main sweep
# -------------------------
if __name__ == "__main__":
    print("Beginning evaluation sweep...")
    summary_rows = []
    for k in K_SHOTS:
        for model_name, trc in MODELS:
            micro, macro = evaluate_model_on_mmlu(
                model_name=model_name,
                trust_remote_code=trc,
                k_shot=k,
                batch_size_context=BATCH_SIZE_CONTEXT,
                print_per_subject=PRINT_PER_SUBJECT,
            )
            summary_rows.append((model_name, k, micro, macro))

    # Pretty summary
    print("\n==== SUMMARY (Micro & Macro Accuracy) ====")
    # Sort by model then k
    summary_rows.sort(key=lambda x: (x[0], x[1]))
    for name, k, micro, macro in summary_rows:
        print(f"{name:40s}  k={k:3d}  micro={micro:.4f}  macro={macro:.4f}")
