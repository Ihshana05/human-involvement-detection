"""
STEP 0 — Dataset Reconstruction
Human Involvement Detection in AI-Assisted Plagiarism

Input : CHEAT_DATASET.csv  (4,514 rows × 5 cols — wide format)
Output: master_dataset.csv (18,056 rows × 4 cols — long format)

Columns in output:
  abstract_id  — links all 4 variants of the same abstract
  label        — 0=ChatGPT, 1=AI-Polished, 2=Human-Fusion, 3=Human
  label_name   — human-readable class name
  text         — raw text (used as-is for ALL feature modules)

Why long format?
  Each row = one text with one label.
  The human reference for L and S is retrieved via abstract_id at feature time.
  Split happens AFTER feature extraction, at abstract_id level (no leakage).
"""

import pandas as pd
import sys

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "CHEAT_DATASET.csv"
OUTPUT_FILE = "master_dataset.csv"

# Map: column name in CSV → (label int, label name)
VARIANT_MAP = {
    "chatgpt"         : (0, "ChatGPT"),
    "ai_polished"     : (1, "AI-Polished"),
    "human_ai_fusion" : (2, "Human-Fusion"),
    "human"           : (3, "Human"),
}

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 0 — Dataset Reconstruction")
print("=" * 60)

try:
    df_raw = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"\n  ERROR: '{INPUT_FILE}' not found.")
    print("  Make sure you uploaded CHEAT_DATASET.csv to Colab.")
    sys.exit(1)

print(f"\n  Loaded : {len(df_raw):,} rows × {len(df_raw.columns)} columns")
print(f"  Columns: {list(df_raw.columns)}")

# ── CHECK COLUMNS ─────────────────────────────────────────────────────────────
missing = [c for c in list(VARIANT_MAP.keys()) + ["abstract_id"]
           if c not in df_raw.columns]
if missing:
    print(f"\n  ERROR: Missing columns: {missing}")
    print("  Available columns:", list(df_raw.columns))
    sys.exit(1)

print(f"\n  All required columns present.")

# ── MELT TO LONG FORMAT ───────────────────────────────────────────────────────
print(f"\n  Reconstructing to long format ...")

rows = []
for _, row in df_raw.iterrows():
    aid = row["abstract_id"]
    for col, (label, label_name) in VARIANT_MAP.items():
        text = str(row[col]).strip() if pd.notna(row[col]) else ""
        rows.append({
            "abstract_id" : aid,
            "label"       : label,
            "label_name"  : label_name,
            "text"        : text,
        })

df_out = pd.DataFrame(rows)

# ── VALIDATE ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Validation")
print("=" * 60)

n_abstracts = df_raw["abstract_id"].nunique()
expected    = n_abstracts * 4
actual      = len(df_out)

print(f"\n  Unique abstracts : {n_abstracts:,}")
print(f"  Expected rows    : {expected:,}  ({n_abstracts} x 4)")
print(f"  Actual rows      : {actual:,}")
print(f"  Row count check  : {'PASS' if actual == expected else 'FAIL !!!'}")

print(f"\n  Class distribution:")
dist = df_out.groupby(["label","label_name"])["abstract_id"].count().reset_index()
dist.columns = ["label","label_name","count"]
print(dist.to_string(index=False))

empty = (df_out["text"] == "").sum()
short = (df_out["text"].str.split().str.len() < 10).sum()
print(f"\n  Empty text cells : {empty}")
print(f"  Texts < 10 words : {short}")

# ── SAMPLE ────────────────────────────────────────────────────────────────────
print(f"\n  Sample — all 4 variants of first abstract:")
first_id = df_out["abstract_id"].iloc[0]
sample   = df_out[df_out["abstract_id"] == first_id][
    ["abstract_id","label","label_name","text"]].copy()
sample["text_preview"] = sample["text"].str[:80] + "..."
print(sample[["abstract_id","label","label_name","text_preview"]].to_string(index=False))

# ── SAVE ──────────────────────────────────────────────────────────────────────
df_out.to_csv(OUTPUT_FILE, index=False)
print(f"\n  Saved: {OUTPUT_FILE}  —  {len(df_out):,} rows x {len(df_out.columns)} columns")

# ── DOWNLOAD (Colab) ──────────────────────────────────────────────────────────
try:
    from google.colab import files
    files.download(OUTPUT_FILE)
    print(f"  Downloaded: {OUTPUT_FILE}")
except ImportError:
    pass  # not in Colab

print("\n" + "=" * 60)
print("STEP 0 COMPLETE")
print("=" * 60)
print(f"""
  Output columns:
    abstract_id  → links all 4 variants
    label        → 0=ChatGPT, 1=AI-Polished, 2=Fusion, 3=Human
    label_name   → human-readable class name
    text         → original text (used for ALL feature modules)

  Note: The 'human' text for each abstract_id is used as the
  REFERENCE in Module L and Module S feature extraction.
  This is what drives 75-85% accuracy — NOT centroid similarity.

  Next: Run step1_preprocessing.py
""")
