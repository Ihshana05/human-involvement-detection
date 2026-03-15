"""
STEP 1 — Text Preprocessing
Human Involvement Detection in AI-Assisted Plagiarism

Input : master_dataset.csv  (18,056 rows × 4 cols)
Output: preprocessed_dataset.csv  (18,056 rows × 6 cols)

New columns added:
  text_raw   — original text, unchanged  → used by Module A (GPT-2), Module S (SBERT)
  text_clean — lowercased, punct removed → used by Module L (N-gram, TF-IDF), Module C (stylometric)

What is applied:
  1. Unicode NFKC normalisation  — fixes ligatures, non-breaking spaces, invisible chars
  2. Whitespace collapse          — removes double spaces, tabs, stray newlines
  3. For text_clean only:
       lowercase + punctuation removal (keeps letters, digits, spaces only)

What is NOT applied (and why):
  - No stemming / lemmatization  — would destroy function-word patterns (Module C)
  - No stopword removal          — function-word freq variance NEEDS stopwords
  - No spell correction          — AI error patterns are valid signals
  - No sentence splitting here   — each module handles its own sentence tokenization
"""

import pandas as pd
import unicodedata
import re
import sys

INPUT_FILE  = "master_dataset.csv"
OUTPUT_FILE = "preprocessed_dataset.csv"

# ── PREPROCESSING FUNCTIONS ───────────────────────────────────────────────────

def normalise_unicode(text: str) -> str:
    """
    NFKC normalisation:
      - Decomposes ligatures:   ﬁ → fi,  ﬂ → fl
      - Normalises spaces:      \u00a0 (non-breaking) → regular space
      - Removes zero-width chars: \u200b, \u200c, \ufeff
      - Collapses composite characters to single codepoints
    """
    text = unicodedata.normalize("NFKC", text)
    # Remove zero-width and invisible characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    return text


def collapse_whitespace(text: str) -> str:
    """
    Replaces all whitespace sequences (tabs, newlines, multiple spaces)
    with a single space and strips leading/trailing whitespace.
    Also fixes missing space after sentence-ending punctuation.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
    return text.strip()


def make_clean(text: str) -> str:
    """
    Lowercase + remove punctuation.
    Keeps: letters (all scripts), digits, spaces.
    Removes: all punctuation, special characters.
    Does NOT stem, lemmatize, or remove stopwords.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)        # collapse spaces again
    return text.strip()


def preprocess_text(raw_text: str):
    """
    Returns (text_raw, text_clean) tuple.
    text_raw  : unicode-fixed + whitespace-collapsed only
    text_clean: text_raw → lowercase → punct removed
    """
    if not isinstance(raw_text, str) or raw_text.strip() == "":
        return "", ""

    # Both versions start with unicode + whitespace fix
    base = collapse_whitespace(normalise_unicode(raw_text))

    text_raw   = base
    text_clean = make_clean(base)

    return text_raw, text_clean


# ── MAIN ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Text Preprocessing")
print("=" * 60)

# Load
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"\n  ERROR: '{INPUT_FILE}' not found.")
    print("  Upload master_dataset.csv first.")
    sys.exit(1)

print(f"\n  Loaded : {len(df):,} rows × {len(df.columns)} columns")
print(f"  Columns: {list(df.columns)}")

# Process
print(f"\n  Preprocessing {len(df):,} texts ...")
print("  (unicode fix → whitespace collapse → clean version)")

raw_texts   = []
clean_texts = []

for i, text in enumerate(df["text"]):
    raw, clean = preprocess_text(str(text))
    raw_texts.append(raw)
    clean_texts.append(clean)

    if (i + 1) % 3000 == 0 or (i + 1) == len(df):
        pct = (i + 1) / len(df) * 100
        print(f"  {i+1:,} / {len(df):,}  ({pct:.1f}%)")

df["text_raw"]   = raw_texts
df["text_clean"] = clean_texts

# ── VALIDATION ────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("Validation")
print("=" * 60)

print(f"\n  Rows          : {len(df):,}  (expected 18,056)")
print(f"  Columns       : {list(df.columns)}")
print(f"  Empty text_raw   : {(df['text_raw']   == '').sum()}")
print(f"  Empty text_clean : {(df['text_clean'] == '').sum()}")

# Word count comparison per class
df["raw_wc"]   = df["text_raw"].str.split().str.len()
df["clean_wc"] = df["text_clean"].str.split().str.len()

wc = df.groupby("label_name")[["raw_wc","clean_wc"]].mean().round(1)
print(f"\n  Word count comparison (mean per class):")
print(wc.to_string())
print("  (clean_wc slightly lower than raw_wc = correct)")

# Show example
print(f"\n  Example — abstract_id {df['abstract_id'].iloc[0]}  (Human row):")
human_row = df[(df["abstract_id"] == df["abstract_id"].iloc[0]) & (df["label"] == 3)].iloc[0]
print(f"  RAW   : {human_row['text_raw'][:120]}...")
print(f"  CLEAN : {human_row['text_clean'][:120]}...")

# Drop helper columns
df.drop(columns=["raw_wc","clean_wc"], inplace=True)

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n  Saved: {OUTPUT_FILE}  —  {len(df):,} rows x {len(df.columns)} columns")

try:
    from google.colab import files
    files.download(OUTPUT_FILE)
    print(f"  Downloaded: {OUTPUT_FILE}")
except ImportError:
    pass

print(f"\n{'=' * 60}")
print("STEP 1 COMPLETE")
print("=" * 60)
print(f"""
  Output columns:
    abstract_id  → links all 4 variants
    label        → 0/1/2/3
    label_name   → class name
    text         → original raw (backup)
    text_raw     → unicode-fixed, whitespace-clean  → GPT-2, SBERT
    text_clean   → lowercase, punct-free            → N-gram, TF-IDF, stylometric counts

  Next: Run step2_split.py
  (Split BEFORE feature extraction — but we need the human reference
   available in every split, so split happens at abstract_id level)
""")
