"""
STEP 3 — Module L : Lexical Similarity
Human Involvement Detection in AI-Assisted Plagiarism

Input : train_dataset.csv, val_dataset.csv, test_dataset.csv
Output: train_L.csv, val_L.csv, test_L.csv

New column added: L
  L = 0.5 * bigram_jaccard  +  0.5 * tfidf_cosine
  Range: 0.0 (no lexical overlap) to 1.0 (identical)

How it works:
  For each row, compare text_clean against the human reference
  of the SAME abstract_id (retrieved from the same split).

  Human row       -> compared against itself           -> L = 1.0
  Human-Fusion    -> compared against human original   -> L ~ 0.60-0.75
  AI-Polished     -> compared against human original   -> L ~ 0.35-0.50
  ChatGPT         -> compared against human original   -> L ~ 0.02-0.08

TF-IDF:
  Fitted on TRAIN texts only (no leakage).
  Same fitted vectorizer applied to val and test.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import sys

print("=" * 60)
print("STEP 3 — Module L : Lexical Similarity")
print("=" * 60)

# ── LOAD ──────────────────────────────────────────────────────────────────────
for fname in ["train_dataset.csv", "val_dataset.csv", "test_dataset.csv"]:
    try:
        open(fname)
    except FileNotFoundError:
        print(f"\n  ERROR: '{fname}' not found. Upload all 3 split files.")
        sys.exit(1)

df_train = pd.read_csv("train_dataset.csv")
df_val   = pd.read_csv("val_dataset.csv")
df_test  = pd.read_csv("test_dataset.csv")

print(f"\n  Train : {len(df_train):,} | Val : {len(df_val):,} | Test : {len(df_test):,}")

# ── BUILD HUMAN REFERENCE LOOKUP ──────────────────────────────────────────────
# For each split, build a dict: abstract_id -> human text_clean
def build_human_lookup(df):
    human_rows = df[df["label"] == 3][["abstract_id", "text_clean"]].copy()
    return dict(zip(human_rows["abstract_id"], human_rows["text_clean"]))

human_ref_train = build_human_lookup(df_train)
human_ref_val   = build_human_lookup(df_val)
human_ref_test  = build_human_lookup(df_test)

print(f"\n  Human references available:")
print(f"    Train : {len(human_ref_train):,} abstracts")
print(f"    Val   : {len(human_ref_val):,}  abstracts")
print(f"    Test  : {len(human_ref_test):,}  abstracts")

# ── FIT TF-IDF ON TRAIN ONLY ──────────────────────────────────────────────────
print(f"\n  Fitting TF-IDF on TRAIN texts only ...")
tfidf = TfidfVectorizer(
    max_features = 15000,   # top 15k terms — enough for IEEE abstracts
    ngram_range  = (1, 2),  # unigrams + bigrams in TF-IDF
    min_df       = 2,       # ignore terms appearing in only 1 doc
    sublinear_tf = True,    # log(1+tf) — reduces impact of very frequent terms
)
tfidf.fit(df_train["text_clean"].fillna("").tolist())
print(f"  Vocabulary size : {len(tfidf.vocabulary_):,} features")
print(f"  Fitted on TRAIN only — will apply same vectorizer to val and test.")

# ── BIGRAM JACCARD ────────────────────────────────────────────────────────────
def bigram_jaccard(text1: str, text2: str) -> float:
    """
    Jaccard similarity over word bigram sets.
    Formula: (2 * |intersection|) / (|bigrams_1| + |bigrams_2|)
    Returns 0.0 if both texts are empty.
    """
    words1 = text1.split()
    words2 = text2.split()
    if len(words1) < 2 or len(words2) < 2:
        return 0.0
    bg1 = Counter(zip(words1, words1[1:]))
    bg2 = Counter(zip(words2, words2[1:]))
    # Multiset intersection
    common = sum((bg1 & bg2).values())
    total  = sum(bg1.values()) + sum(bg2.values())
    return (2 * common) / total if total > 0 else 0.0

# ── TF-IDF COSINE ─────────────────────────────────────────────────────────────
def tfidf_cosine_batch(texts_a, texts_b, vectorizer) -> np.ndarray:
    """
    Compute TF-IDF cosine similarity between paired texts.
    texts_a[i] compared against texts_b[i].
    Returns array of shape (n,).
    """
    vecs_a = vectorizer.transform(texts_a)
    vecs_b = vectorizer.transform(texts_b)
    # Row-wise cosine (diagonal of full matrix)
    scores = np.array([
        cosine_similarity(vecs_a[i], vecs_b[i])[0, 0]
        for i in range(vecs_a.shape[0])
    ])
    return scores

# ── COMPUTE L FOR ONE SPLIT ───────────────────────────────────────────────────
def compute_L(df: pd.DataFrame, human_ref: dict, split_name: str) -> pd.DataFrame:
    """
    For every row in df:
      1. Get human reference text for same abstract_id
      2. Compute bigram_jaccard(text_clean, human_ref)
      3. Compute tfidf_cosine(text_clean, human_ref)
      4. L = 0.5 * bigram + 0.5 * tfidf
    """
    print(f"\n{'=' * 60}")
    print(f"Module L — {split_name}  ({len(df):,} rows)")
    print("=" * 60)

    candidates = df["text_clean"].fillna("").tolist()
    references = [
        human_ref.get(aid, "")
        for aid in df["abstract_id"]
    ]

    # Check no missing references
    missing = sum(1 for r in references if r == "")
    if missing > 0:
        print(f"  WARNING: {missing} rows have no human reference!")

    # Bigram Jaccard — row by row (fast enough for 12k texts)
    print(f"  Computing bigram Jaccard ...")
    bigram_scores = []
    for i, (cand, ref) in enumerate(zip(candidates, references)):
        bigram_scores.append(bigram_jaccard(cand, ref))
        if (i + 1) % 3000 == 0 or (i + 1) == len(df):
            print(f"    {i+1:,} / {len(df):,}  ({(i+1)/len(df)*100:.1f}%)")

    # TF-IDF cosine — batch
    print(f"  Computing TF-IDF cosine ...")
    tfidf_scores = tfidf_cosine_batch(candidates, references, tfidf)

    # Final L
    bigram_arr = np.array(bigram_scores)
    L = 0.5 * bigram_arr + 0.5 * tfidf_scores

    df_out = df.copy()
    df_out["bigram_jaccard"] = bigram_arr.round(6)
    df_out["tfidf_cosine"]   = tfidf_scores.round(6)
    df_out["L"]              = L.round(6)

    # Report
    print(f"\n  Feature means by class ({split_name}):")
    means = df_out.groupby("label_name")[["bigram_jaccard","tfidf_cosine","L"]].mean().round(4)
    print(means.to_string())

    null_count = df_out["L"].isnull().sum()
    print(f"\n  Null values in L : {null_count}  (expected 0)")

    return df_out

# ── RUN ALL SPLITS ────────────────────────────────────────────────────────────
print(f"\n  Processing all 3 splits ...")
train_L = compute_L(df_train, human_ref_train, "TRAIN")
val_L   = compute_L(df_val,   human_ref_val,   "VAL")
test_L  = compute_L(df_test,  human_ref_test,  "TEST")

# ── SAVE ──────────────────────────────────────────────────────────────────────
print(f"\n  Saving ...")
train_L.to_csv("train_L.csv", index=False)
val_L.to_csv("val_L.csv",     index=False)
test_L.to_csv("test_L.csv",   index=False)
print(f"    train_L.csv -> {len(train_L):,} rows")
print(f"    val_L.csv   -> {len(val_L):,}  rows")
print(f"    test_L.csv  -> {len(test_L):,}  rows")

try:
    from google.colab import files
    for fname in ["train_L.csv","val_L.csv","test_L.csv"]:
        files.download(fname)
    print(f"  All 3 files downloaded.")
except ImportError:
    pass

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("STEP 3 COMPLETE — Module L")
print("=" * 60)
print(f"""
  Features added : bigram_jaccard, tfidf_cosine, L
  TF-IDF fitted  : TRAIN only (no leakage)
  L formula      : 0.5 * bigram_jaccard + 0.5 * tfidf_cosine

  Expected L means (check your output):
    Human          ~ 1.000  (compared against itself)
    Human-Fusion   ~ 0.60-0.75  (shared human sentences)
    AI-Polished    ~ 0.35-0.50  (partially rewritten)
    ChatGPT        ~ 0.02-0.08  (completely different vocab)

  If you see this pattern -> L feature is working correctly.

  Next: Run step4_module_S.py  (SBERT semantic similarity)
  Ensure GPU is ON — SBERT encoding is faster with GPU.
""")
