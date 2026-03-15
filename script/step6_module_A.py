"""
STEP 6 — Module A : AI Signal Features
Human Involvement Detection in AI-Assisted Plagiarism

Input : train_C.csv, val_C.csv, test_C.csv
Output: train_A.csv, val_A.csv, test_A.csv

4 Features added (all intrinsic — no reference needed):
  burstiness     : word frequency variability (σ-μ)/(σ+μ)
  rep_score      : 1 - bigram repetition rate
  log_prob_score : GPT-2 log-probability per token (inverted — higher = more human)
  detector_score : weighted combination of log_prob + burstiness

  Final A = mean(burstiness_norm, rep_score_norm,
                 log_prob_score_norm, detector_score_norm)
  All sub-scores normalised to [0,1] using train min/max.

Expected pattern (higher = more human-like):
  Human > Human-Fusion > AI-Polished > ChatGPT

Key fix vs previous attempt:
  log_prob is computed PER TEXT (not per batch) so every row gets
  a unique value. The previous implementation computed one loss per
  batch of 32 and copied it to all 32 rows — that was wrong.
"""

import pandas as pd
import numpy as np
import re
import sys
import math
from collections import Counter

print("=" * 60)
print("STEP 6 — Module A : AI Signal Features")
print("=" * 60)

# ── INSTALL ───────────────────────────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "transformers", "torch", "-q"], check=True)

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
for fname in ["train_C.csv", "val_C.csv", "test_C.csv"]:
    try:
        open(fname)
    except FileNotFoundError:
        print(f"\n  ERROR: '{fname}' not found. Upload all 3 C files.")
        sys.exit(1)

df_train = pd.read_csv("train_C.csv")
df_val   = pd.read_csv("val_C.csv")
df_test  = pd.read_csv("test_C.csv")
print(f"\n  Train : {len(df_train):,} | Val : {len(df_val):,} | Test : {len(df_test):,}")

# ── LOAD GPT-2 ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device : {device}")
print(f"  Loading GPT-2 ...")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2.eval()
print(f"  GPT-2 loaded on {device}")

# ── FEATURE FUNCTIONS ─────────────────────────────────────────────────────────

def burstiness(text: str) -> float:
    """
    Burstiness of word usage: (std - mean) / (std + mean)
    Applied to word frequency counts across the text.
    Range: -1 (perfectly uniform) to +1 (maximally bursty).
    Higher = more human-like variation.
    We shift to [0, 1]: score = (burstiness + 1) / 2
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 10:
        return 0.5
    counts = np.array(list(Counter(words).values()), dtype=float)
    mu  = counts.mean()
    sig = counts.std()
    if mu + sig == 0:
        return 0.5
    B = (sig - mu) / (sig + mu)
    return float((B + 1) / 2)   # shift to [0, 1]


def repetition_score(text: str) -> float:
    """
    1 - bigram repetition rate.
    rep_rate = repeated bigrams / total bigrams
    Higher rep_rate = more repetitive = more AI-like.
    We invert: rep_score = 1 - rep_rate  (higher = more human).
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 4:
        return 0.5
    bigrams = list(zip(words, words[1:]))
    total   = len(bigrams)
    counts  = Counter(bigrams)
    repeated = sum(c for c in counts.values() if c > 1)
    rep_rate = repeated / total if total > 0 else 0.0
    return float(1.0 - rep_rate)   # invert


def log_prob_per_text(text: str) -> float:
    """
    GPT-2 mean log-probability per token for the WHOLE text.
    Returns negative loss (= log probability).
    More negative = human (GPT-2 finds it surprising).
    Less negative = AI   (GPT-2 finds it predictable).
    Computed PER TEXT — every row gets a unique value.
    """
    if not text or len(text.split()) < 5:
        return -5.0   # default for very short texts

    enc = tokenizer(
        text,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 256
    ).to(device)

    with torch.no_grad():
        loss = gpt2(**enc, labels=enc["input_ids"]).loss

    return float(-loss.item())   # negative loss = log probability


def detector_score(log_prob: float, burst: float) -> float:
    """
    Simple ensemble detector:
    Combines GPT-2 log-probability signal and burstiness signal.
    Both are already on a scale where higher = more human.
    Uses sigmoid to map log_prob to [0,1]:
      sigmoid(x) near 0 = very negative log_prob = human
      sigmoid(x) near 1 = less negative log_prob = AI
    We invert: human_like = 1 - sigmoid(log_prob + offset)

    detector_score = 0.6 * human_like_logprob + 0.4 * burst
    """
    # Shift log_prob by typical mean (~-3.5) to centre the sigmoid
    sigmoid_val = 1.0 / (1.0 + math.exp(-(log_prob + 3.5)))
    human_like  = 1.0 - sigmoid_val   # invert: more negative = higher score
    return float(0.6 * human_like + 0.4 * burst)


# ── NORMALISE ─────────────────────────────────────────────────────────────────
def minmax_norm(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    rng = vmax - vmin
    if rng == 0:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - vmin) / rng, 0.0, 1.0)


# ── EXTRACT FOR ONE SPLIT ─────────────────────────────────────────────────────
def extract_A_features(df: pd.DataFrame, split_name: str) -> tuple:
    print(f"\n{'=' * 60}")
    print(f"Module A — {split_name}  ({len(df):,} rows)")
    print("=" * 60)

    bursts, reps, lps, dets = [], [], [], []

    for i, text in enumerate(df["text_raw"].fillna("")):
        b  = burstiness(text)
        r  = repetition_score(text)
        lp = log_prob_per_text(text)
        d  = detector_score(lp, b)

        bursts.append(b)
        reps.append(r)
        lps.append(lp)
        dets.append(d)

        if (i + 1) % 1000 == 0 or (i + 1) == len(df):
            pct = (i + 1) / len(df) * 100
            print(f"  {i+1:,} / {len(df):,}  ({pct:.1f}%)")

    return (
        np.array(bursts),
        np.array(reps),
        np.array(lps),
        np.array(dets)
    )


# ── RUN ALL SPLITS ────────────────────────────────────────────────────────────
print(f"\n  Extracting features (~15-20 min on GPU) ...")
b_tr, r_tr, lp_tr, d_tr = extract_A_features(df_train, "TRAIN")
b_va, r_va, lp_va, d_va = extract_A_features(df_val,   "VAL")
b_te, r_te, lp_te, d_te = extract_A_features(df_test,  "TEST")

# ── FIT NORMALISATION ON TRAIN ────────────────────────────────────────────────
print(f"\n  Fitting min-max normalisation on TRAIN ...")
norms = {
    "b"  : (b_tr.min(),  b_tr.max()),
    "r"  : (r_tr.min(),  r_tr.max()),
    "lp" : (lp_tr.min(), lp_tr.max()),
    "d"  : (d_tr.min(),  d_tr.max()),
}
for k, (lo, hi) in norms.items():
    print(f"  {k:<4} range : [{lo:.4f}, {hi:.4f}]")

# ── BUILD OUTPUT DATAFRAMES ───────────────────────────────────────────────────
def build_A_df(df, b, r, lp, d, norms, split_name):
    b_n  = minmax_norm(b,  *norms["b"])
    r_n  = minmax_norm(r,  *norms["r"])
    lp_n = minmax_norm(lp, *norms["lp"])
    d_n  = minmax_norm(d,  *norms["d"])
    A    = (b_n + r_n + lp_n + d_n) / 4.0

    df_out = df.copy()
    df_out["burstiness"]     = b.round(6)
    df_out["rep_score"]      = r.round(6)
    df_out["log_prob"]       = lp.round(6)
    df_out["detector_score"] = d.round(6)
    df_out["A"]              = A.round(6)

    print(f"\n  Feature means by class ({split_name}):")
    means = df_out.groupby("label_name")[
        ["burstiness","rep_score","log_prob","detector_score","A"]
    ].mean().round(4)
    print(means.to_string())
    print(f"  Null values in A : {df_out['A'].isnull().sum()}  (expected 0)")

    # Verify log_prob uniqueness — must NOT all be same value
    unique_lp = df_out["log_prob"].nunique()
    print(f"  Unique log_prob values : {unique_lp}  (should be close to {len(df_out)})")

    return df_out


print(f"\n  Building output dataframes ...")
train_A = build_A_df(df_train, b_tr, r_tr, lp_tr, d_tr, norms, "TRAIN")
val_A   = build_A_df(df_val,   b_va, r_va, lp_va, d_va, norms, "VAL")
test_A  = build_A_df(df_test,  b_te, r_te, lp_te, d_te, norms, "TEST")

# ── SAVE ──────────────────────────────────────────────────────────────────────
print(f"\n  Saving ...")
train_A.to_csv("train_A.csv", index=False)
val_A.to_csv("val_A.csv",     index=False)
test_A.to_csv("test_A.csv",   index=False)
print(f"    train_A.csv -> {len(train_A):,} rows")
print(f"    val_A.csv   -> {len(val_A):,}  rows")
print(f"    test_A.csv  -> {len(test_A):,}  rows")

try:
    from google.colab import files
    for fname in ["train_A.csv","val_A.csv","test_A.csv"]:
        files.download(fname)
    print(f"  All 3 files downloaded.")
except ImportError:
    pass

print(f"\n{'=' * 60}")
print("STEP 6 COMPLETE — Module A")
print("=" * 60)
print(f"""
  Features added : burstiness, rep_score, log_prob, detector_score, A
  Normalisation  : min-max fitted on TRAIN only
  A formula      : mean(burstiness_norm, rep_score_norm,
                        log_prob_score_norm, detector_score_norm)

  Key check: unique log_prob values should be close to row count.
  If all log_prob values are the same → something went wrong.

  Expected A ordering: Human > Human-Fusion > AI-Polished > ChatGPT

  Next: Run step7_combine_and_train.py
  Combines L + S + C + A  →  trains Multinomial Logistic Regression
  →  HIRS = P(Human)  →  evaluates on val set
  No GPU needed for training step.
""")
