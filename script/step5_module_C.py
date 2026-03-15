"""
STEP 5 — Module C : Stylometric Consistency
Human Involvement Detection in AI-Assisted Plagiarism

Input : train_S.csv, val_S.csv, test_S.csv
Output: train_C.csv, val_C.csv, test_C.csv

4 Features added (all intrinsic — no reference needed):
  fw_var      : function-word frequency variance across sentences
  sl_entropy  : Shannon entropy of sentence length distribution
  pos_drift   : POS tag ratio drift across text segments
  perp_fluct  : GPT-2 per-sentence perplexity fluctuation (std dev)

  Final C = mean(fw_var_norm, sl_entropy_norm, pos_drift_norm, perp_fluct_norm)
  All sub-scores normalised to [0,1] using train min/max before combining.

Expected pattern (higher = more human-like):
  Human        > Human-Fusion > AI-Polished > ChatGPT

Why these features work:
  AI models generate text with uniform style throughout — consistent
  sentence lengths, stable POS patterns, predictable token sequences.
  Human writing is irregular — varies in rhythm, structure, and phrasing.
  These features capture that irregularity directly.

GPU recommended for perplexity fluctuation (GPT-2 per sentence).
"""

import pandas as pd
import numpy as np
import re
import sys
import math

print("=" * 60)
print("STEP 5 — Module C : Stylometric Consistency")
print("=" * 60)

# ── INSTALL DEPENDENCIES ──────────────────────────────────────────────────────
print("\n  Installing dependencies ...")
import subprocess
subprocess.run(["pip", "install", "nltk", "transformers", "torch", "-q"], check=True)

import nltk
for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng", "stopwords"]:
    nltk.download(pkg, quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk import pos_tag
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
for fname in ["train_S.csv", "val_S.csv", "test_S.csv"]:
    try:
        open(fname)
    except FileNotFoundError:
        print(f"\n  ERROR: '{fname}' not found.")
        sys.exit(1)

df_train = pd.read_csv("train_S.csv")
df_val   = pd.read_csv("val_S.csv")
df_test  = pd.read_csv("test_S.csv")
print(f"\n  Train : {len(df_train):,} | Val : {len(df_val):,} | Test : {len(df_test):,}")

# ── LOAD GPT-2 ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device : {device}")
print(f"  Loading GPT-2 for perplexity fluctuation ...")

tokenizer_gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model_gpt2.eval()
print(f"  GPT-2 loaded on {device}")

# ── FUNCTION WORDS ────────────────────────────────────────────────────────────
FUNCTION_WORDS = set([
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","shall","should",
    "may","might","must","can","could","this","that","these","those",
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","of","in","to","for","on",
    "with","at","by","from","as","into","about","than","but","and","or",
    "not","if","so","up","out","what","which","who","all","when","there"
])

# ── FEATURE FUNCTIONS ─────────────────────────────────────────────────────────

def function_word_variance(text: str) -> float:
    """
    Variance of function-word ratio across sentences.
    Low = AI (uniform usage), High = human (varied usage).
    """
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return 0.0
    ratios = []
    for sent in sentences:
        words = word_tokenize(sent.lower())
        words = [w for w in words if w.isalpha()]
        if len(words) == 0:
            continue
        fw_count = sum(1 for w in words if w in FUNCTION_WORDS)
        ratios.append(fw_count / len(words))
    return float(np.var(ratios)) if len(ratios) >= 2 else 0.0


def sentence_length_entropy(text: str) -> float:
    """
    Shannon entropy of sentence length distribution.
    Low = AI (uniform lengths), High = human (varied lengths).
    """
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return 0.0
    lengths = [len(word_tokenize(s)) for s in sentences]
    # Bin into 5-word buckets
    bins = {}
    for l in lengths:
        b = (l // 5) * 5
        bins[b] = bins.get(b, 0) + 1
    total = sum(bins.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in bins.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def pos_drift(text: str) -> float:
    """
    Standard deviation of noun/verb/adj ratio across 3 text segments.
    Low = AI (rigid POS pattern), High = human (varied structure).
    """
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return 0.0
    # Split into 3 equal segments
    seg_size = len(sentences) // 3
    if seg_size == 0:
        return 0.0
    segments = [
        sentences[:seg_size],
        sentences[seg_size:2*seg_size],
        sentences[2*seg_size:]
    ]
    ratios = []
    for seg in segments:
        seg_text = " ".join(seg)
        words    = word_tokenize(seg_text)
        if len(words) == 0:
            continue
        tags   = pos_tag(words)
        content = sum(1 for _, t in tags if t.startswith(('NN','VB','JJ')))
        ratios.append(content / len(words))
    return float(np.std(ratios)) if len(ratios) >= 2 else 0.0


def perplexity_fluctuation(text: str) -> float:
    """
    Standard deviation of per-sentence GPT-2 perplexity.
    Low = AI (uniformly predictable), High = human (varied predictability).
    Computed per sentence to capture local variation.
    """
    sentences = sent_tokenize(text)
    sentences = [s for s in sentences if len(s.split()) >= 4]
    if len(sentences) < 3:
        return 0.0

    perplexities = []
    for sent in sentences:
        try:
            enc = tokenizer_gpt2(
                sent,
                return_tensors  = "pt",
                truncation      = True,
                max_length      = 128
            ).to(device)
            with torch.no_grad():
                loss = model_gpt2(**enc, labels=enc["input_ids"]).loss
            perp = math.exp(min(loss.item(), 20))   # cap at exp(20) to avoid inf
            perplexities.append(perp)
        except Exception:
            continue

    return float(np.std(perplexities)) if len(perplexities) >= 2 else 0.0


# ── NORMALISE USING TRAIN STATS ───────────────────────────────────────────────
def minmax_norm(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Min-max normalise to [0,1]. Fitted values come from train."""
    rng = vmax - vmin
    if rng == 0:
        return np.zeros_like(values)
    return np.clip((values - vmin) / rng, 0.0, 1.0)


# ── COMPUTE C FOR ONE SPLIT ───────────────────────────────────────────────────
def extract_features(df: pd.DataFrame, split_name: str) -> tuple:
    """Extract raw sub-scores for all rows. Returns arrays."""
    print(f"\n{'=' * 60}")
    print(f"Module C — {split_name}  ({len(df):,} rows)")
    print("=" * 60)

    fw_vars, sl_entropies, pos_drifts, perp_flucts = [], [], [], []

    for i, text in enumerate(df["text_raw"].fillna("")):
        fw_vars.append(function_word_variance(text))
        sl_entropies.append(sentence_length_entropy(text))
        pos_drifts.append(pos_drift(text))
        perp_flucts.append(perplexity_fluctuation(text))

        if (i + 1) % 1000 == 0 or (i + 1) == len(df):
            pct = (i + 1) / len(df) * 100
            print(f"  {i+1:,} / {len(df):,}  ({pct:.1f}%)")

    return (
        np.array(fw_vars),
        np.array(sl_entropies),
        np.array(pos_drifts),
        np.array(perp_flucts)
    )

# ── EXTRACT RAW FEATURES ──────────────────────────────────────────────────────
print(f"\n  Extracting features (this takes ~20-30 min on GPU) ...")
print(f"  Tip: Module C processes sentences individually — no batch shortcut.")

fw_train, sl_train, pd_train, pf_train = extract_features(df_train, "TRAIN")
fw_val,   sl_val,   pd_val,   pf_val   = extract_features(df_val,   "VAL")
fw_test,  sl_test,  pd_test,  pf_test  = extract_features(df_test,  "TEST")

# ── FIT NORMALISATION ON TRAIN ────────────────────────────────────────────────
print(f"\n  Fitting min-max normalisation on TRAIN ...")
norms = {
    "fw"   : (fw_train.min(),  fw_train.max()),
    "sl"   : (sl_train.min(),  sl_train.max()),
    "pd"   : (pd_train.min(),  pd_train.max()),
    "pf"   : (pf_train.min(),  pf_train.max()),
}
print(f"  fw_var      range : [{norms['fw'][0]:.6f}, {norms['fw'][1]:.6f}]")
print(f"  sl_entropy  range : [{norms['sl'][0]:.4f}, {norms['sl'][1]:.4f}]")
print(f"  pos_drift   range : [{norms['pd'][0]:.6f}, {norms['pd'][1]:.6f}]")
print(f"  perp_fluct  range : [{norms['pf'][0]:.4f}, {norms['pf'][1]:.4f}]")

# ── BUILD OUTPUT DATAFRAMES ───────────────────────────────────────────────────
def build_C_df(df, fw, sl, pd_arr, pf, norms, split_name):
    fw_n  = minmax_norm(fw,     *norms["fw"])
    sl_n  = minmax_norm(sl,     *norms["sl"])
    pd_n  = minmax_norm(pd_arr, *norms["pd"])
    pf_n  = minmax_norm(pf,     *norms["pf"])
    C     = (fw_n + sl_n + pd_n + pf_n) / 4.0

    df_out = df.copy()
    df_out["fw_var"]     = fw.round(6)
    df_out["sl_entropy"] = sl.round(6)
    df_out["pos_drift"]  = pd_arr.round(6)
    df_out["perp_fluct"] = pf.round(4)
    df_out["C"]          = C.round(6)

    print(f"\n  Feature means by class ({split_name}):")
    means = df_out.groupby("label_name")[["fw_var","sl_entropy","pos_drift","perp_fluct","C"]].mean().round(4)
    print(means.to_string())
    print(f"  Null values in C : {df_out['C'].isnull().sum()}  (expected 0)")
    return df_out

print(f"\n  Building output dataframes ...")
train_C = build_C_df(df_train, fw_train, sl_train, pd_train, pf_train, norms, "TRAIN")
val_C   = build_C_df(df_val,   fw_val,   sl_val,   pd_val,   pf_val,   norms, "VAL")
test_C  = build_C_df(df_test,  fw_test,  sl_test,  pd_test,  pf_test,  norms, "TEST")

# ── SAVE ──────────────────────────────────────────────────────────────────────
print(f"\n  Saving ...")
train_C.to_csv("train_C.csv", index=False)
val_C.to_csv("val_C.csv",     index=False)
test_C.to_csv("test_C.csv",   index=False)
print(f"    train_C.csv -> {len(train_C):,} rows")
print(f"    val_C.csv   -> {len(val_C):,}  rows")
print(f"    test_C.csv  -> {len(test_C):,}  rows")

try:
    from google.colab import files
    for fname in ["train_C.csv","val_C.csv","test_C.csv"]:
        files.download(fname)
    print(f"  All 3 files downloaded.")
except ImportError:
    pass

print(f"\n{'=' * 60}")
print("STEP 5 COMPLETE — Module C")
print("=" * 60)
print(f"""
  Features added : fw_var, sl_entropy, pos_drift, perp_fluct, C
  Normalisation  : min-max fitted on TRAIN only
  C formula      : mean(fw_var_norm, sl_entropy_norm,
                        pos_drift_norm, perp_fluct_norm)

  Expected C ordering: Human > Human-Fusion > AI-Polished > ChatGPT

  Next: Run step6_module_A.py  (AI Signal features — burstiness,
        repetition rate, log-probability, ensemble detector)
  GPU needed for log-probability (GPT-2 per text).
""")
