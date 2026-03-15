"""
STEP 4 — Module S : Semantic Similarity (SBERT)
Human Involvement Detection in AI-Assisted Plagiarism

Input : train_L.csv, val_L.csv, test_L.csv
Output: train_S.csv, val_S.csv, test_S.csv

New column added: S
  S = cosine_similarity(SBERT(candidate), SBERT(human_reference))
  Range: 0.0 to 1.0

Model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional sentence embeddings
  - Fast and accurate for semantic similarity
  - Trained on natural text — use text_raw (NOT text_clean)

Why S complements L:
  L measures vocabulary overlap (surface level)
  S measures meaning preservation (semantic level)

  ChatGPT rewrites both surface AND meaning  → low L, moderate S
  AI-Polished rewrites surface, keeps meaning → low-mid L, high S
  Human-Fusion keeps surface AND meaning      → high L, high S
  Human is the reference                      → L=1.0, S=1.0

  The L vs S difference is what separates AI-Polished from ChatGPT.

Expected S means:
  Human          ~ 1.000
  Human-Fusion   ~ 0.92-0.97
  AI-Polished    ~ 0.85-0.92
  ChatGPT        ~ 0.65-0.78

IMPORTANT: Ensure GPU runtime is ON in Colab for this step.
  Runtime -> Change runtime type -> T4 GPU
"""

import pandas as pd
import numpy as np
import sys

print("=" * 60)
print("STEP 4 — Module S : Semantic Similarity (SBERT)")
print("=" * 60)

# ── INSTALL + LOAD ─────────────────────────────────────────────────────────────
print("\n  Installing sentence-transformers ...")
import subprocess
subprocess.run(["pip", "install", "sentence-transformers", "-q"], check=True)

from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device : {device}")
if device == "cpu":
    print("  WARNING: GPU not detected. This will take ~45 minutes.")
    print("  Recommended: Runtime -> Change runtime type -> T4 GPU")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
for fname in ["train_L.csv", "val_L.csv", "test_L.csv"]:
    try:
        open(fname)
    except FileNotFoundError:
        print(f"\n  ERROR: '{fname}' not found.")
        sys.exit(1)

df_train = pd.read_csv("train_L.csv")
df_val   = pd.read_csv("val_L.csv")
df_test  = pd.read_csv("test_L.csv")

print(f"\n  Train : {len(df_train):,} | Val : {len(df_val):,} | Test : {len(df_test):,}")

# ── LOAD SBERT MODEL ──────────────────────────────────────────────────────────
print(f"\n  Loading SBERT (all-MiniLM-L6-v2) ...")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print(f"  Model loaded on {device}")

# ── BUILD HUMAN REFERENCE LOOKUP ──────────────────────────────────────────────
def build_human_lookup(df):
    human_rows = df[df["label"] == 3][["abstract_id", "text_raw"]].copy()
    return dict(zip(human_rows["abstract_id"], human_rows["text_raw"]))

human_ref_train = build_human_lookup(df_train)
human_ref_val   = build_human_lookup(df_val)
human_ref_test  = build_human_lookup(df_test)

# ── COSINE SIMILARITY ─────────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

# ── COMPUTE S FOR ONE SPLIT ───────────────────────────────────────────────────
def compute_S(df: pd.DataFrame, human_ref: dict, split_name: str) -> pd.DataFrame:
    """
    For every row:
      1. Get human reference text (text_raw) for same abstract_id
      2. Encode both candidate and reference with SBERT
      3. S = cosine_similarity(candidate_emb, reference_emb)
    """
    print(f"\n{'=' * 60}")
    print(f"Module S — {split_name}  ({len(df):,} rows)")
    print("=" * 60)

    candidates = df["text_raw"].fillna("").tolist()
    references = [human_ref.get(aid, "") for aid in df["abstract_id"]]

    missing = sum(1 for r in references if r == "")
    if missing > 0:
        print(f"  WARNING: {missing} rows have no human reference!")

    # Encode all candidates
    print(f"  Encoding candidate texts ...")
    cand_embs = model.encode(
        candidates,
        batch_size     = 64,
        show_progress_bar = True,
        convert_to_numpy  = True,
        normalize_embeddings = True   # pre-normalise for cosine = dot product
    )

    # Encode all references
    print(f"  Encoding reference (human) texts ...")
    ref_embs = model.encode(
        references,
        batch_size     = 64,
        show_progress_bar = True,
        convert_to_numpy  = True,
        normalize_embeddings = True
    )

    # Cosine similarity — since embeddings are normalised, dot product = cosine
    print(f"  Computing cosine similarities ...")
    S_scores = np.sum(cand_embs * ref_embs, axis=1)  # element-wise dot product
    S_scores = np.clip(S_scores, 0.0, 1.0)           # clip to [0,1]

    df_out = df.copy()
    df_out["S"] = S_scores.round(6)

    # Report
    print(f"\n  Feature means by class ({split_name}):")
    means = df_out.groupby("label_name")["S"].mean().round(4)
    print(means.to_string())

    null_count = df_out["S"].isnull().sum()
    print(f"\n  Null values in S : {null_count}  (expected 0)")

    return df_out

# ── RUN ALL SPLITS ────────────────────────────────────────────────────────────
print(f"\n  Processing all 3 splits ...")
train_S = compute_S(df_train, human_ref_train, "TRAIN")
val_S   = compute_S(df_val,   human_ref_val,   "VAL")
test_S  = compute_S(df_test,  human_ref_test,  "TEST")

# ── SAVE ──────────────────────────────────────────────────────────────────────
print(f"\n  Saving ...")
train_S.to_csv("train_S.csv", index=False)
val_S.to_csv("val_S.csv",     index=False)
test_S.to_csv("test_S.csv",   index=False)
print(f"    train_S.csv -> {len(train_S):,} rows")
print(f"    val_S.csv   -> {len(val_S):,}  rows")
print(f"    test_S.csv  -> {len(test_S):,}  rows")

try:
    from google.colab import files
    for fname in ["train_S.csv","val_S.csv","test_S.csv"]:
        files.download(fname)
    print(f"  All 3 files downloaded.")
except ImportError:
    pass

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("STEP 4 COMPLETE — Module S")
print("=" * 60)
print(f"""
  Feature added  : S  (SBERT cosine similarity vs human reference)
  Model          : all-MiniLM-L6-v2 (384-dim sentence embeddings)
  Input used     : text_raw (natural text — not lowercased)
  Reference      : human text of same abstract_id

  Combined with L:
    L (lexical)  separates ChatGPT from others strongly
    S (semantic) separates AI-Polished from ChatGPT
    Together L+S give very clean 4-class separation

  Next: Run step5_module_C.py  (Stylometric features)
  No GPU needed for Module C — runs on CPU.
""")
