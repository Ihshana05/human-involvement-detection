"""
STEP 2 — Dataset Split (70 / 15 / 15)
Human Involvement Detection in AI-Assisted Plagiarism

Input : preprocessed_dataset.csv  (18,056 rows × 6 cols)
Output: train_dataset.csv  (12,636 rows — 70%)
        val_dataset.csv    ( 2,708 rows — 15%)
        test_dataset.csv   ( 2,712 rows — 15%)

Split strategy: at abstract_id level (NOT row level)
  - All 4 variants of the same abstract go to the same split
  - Guarantees: human reference always in same split as candidate
  - Guarantees: zero leakage (test abstracts never seen during training)
  - Seed: 42 (fixed — same split every run)
"""

import pandas as pd
import numpy as np
import sys

INPUT_FILE = "preprocessed_dataset.csv"
SEED       = 42

print("=" * 60)
print("STEP 2 — Dataset Split (70 / 15 / 15)")
print("=" * 60)

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"\n  ERROR: '{INPUT_FILE}' not found.")
    sys.exit(1)

print(f"\n  Loaded : {len(df):,} rows x {len(df.columns)} columns")

all_ids = df["abstract_id"].unique()
n       = len(all_ids)
print(f"\n  Unique abstract_ids : {n:,}")
print(f"  Each has 4 variants  -> {n*4:,} total rows")

rng      = np.random.default_rng(SEED)
shuffled = rng.permutation(all_ids)

n_train  = int(n * 0.70)
n_val    = int(n * 0.15)
n_test   = n - n_train - n_val

train_ids = set(shuffled[:n_train])
val_ids   = set(shuffled[n_train : n_train + n_val])
test_ids  = set(shuffled[n_train + n_val:])

print(f"\n  Abstract-level split:")
print(f"    Train : {len(train_ids):,} abstracts  ({len(train_ids)/n*100:.1f}%)")
print(f"    Val   : {len(val_ids):,}  abstracts  ({len(val_ids)/n*100:.1f}%)")
print(f"    Test  : {len(test_ids):,}  abstracts  ({len(test_ids)/n*100:.1f}%)")

df_train = df[df["abstract_id"].isin(train_ids)].copy().reset_index(drop=True)
df_val   = df[df["abstract_id"].isin(val_ids)].copy().reset_index(drop=True)
df_test  = df[df["abstract_id"].isin(test_ids)].copy().reset_index(drop=True)

print(f"\n  Row-level split:")
print(f"    Train : {len(df_train):,} rows")
print(f"    Val   : {len(df_val):,}  rows")
print(f"    Test  : {len(df_test):,}  rows")
print(f"    Total : {len(df_train)+len(df_val)+len(df_test):,}  (expected 18,056)")

train_set = set(df_train["abstract_id"])
val_set   = set(df_val["abstract_id"])
test_set  = set(df_test["abstract_id"])
tv = len(train_set & val_set)
tt = len(train_set & test_set)
vt = len(val_set   & test_set)

print(f"\n  Leakage check:")
print(f"    Train ^ Val  : {tv}  {'PASS' if tv==0 else 'FAIL!!!'}")
print(f"    Train ^ Test : {tt}  {'PASS' if tt==0 else 'FAIL!!!'}")
print(f"    Val   ^ Test : {vt}  {'PASS' if vt==0 else 'FAIL!!!'}")
print(f"    Overall      : {'PASS' if tv==0 and tt==0 and vt==0 else 'FAIL!!!'}")

print(f"\n  Class distribution per split:")
for name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    print(f"\n    {name}:")
    dist = split_df.groupby(["label","label_name"])["abstract_id"].count()
    for (lbl, lname), cnt in dist.items():
        print(f"      Label {lbl} ({lname:<16}) : {cnt:,}")

print(f"\n  Human reference availability check:")
for name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    human_ids     = set(split_df[split_df["label"] == 3]["abstract_id"])
    candidate_ids = set(split_df[split_df["label"] != 3]["abstract_id"])
    missing       = candidate_ids - human_ids
    status        = "PASS" if len(missing) == 0 else f"FAIL — {len(missing)} missing"
    print(f"    {name}: {status}")

print(f"\n  Saving splits ...")
df_train.to_csv("train_dataset.csv", index=False)
df_val.to_csv("val_dataset.csv",     index=False)
df_test.to_csv("test_dataset.csv",   index=False)
print(f"    train_dataset.csv -> {len(df_train):,} rows")
print(f"    val_dataset.csv   -> {len(df_val):,}  rows")
print(f"    test_dataset.csv  -> {len(df_test):,}  rows")

try:
    from google.colab import files
    for fname in ["train_dataset.csv","val_dataset.csv","test_dataset.csv"]:
        files.download(fname)
    print(f"  All 3 files downloaded.")
except ImportError:
    pass

print(f"\n{'='*60}")
print("STEP 2 COMPLETE")
print("="*60)
print(f"""
  Seed     : {SEED}  (same split every run)
  Strategy : abstract_id level split — zero leakage

  train_dataset.csv  ->  {len(df_train):,} rows  (70%)
  val_dataset.csv    ->  {len(df_val):,}  rows  (15%)
  test_dataset.csv   ->  {len(df_test):,}  rows  (15%)

  Next: Run step3_module_L.py
  (N-gram + TF-IDF — each text vs its own human reference)
""")
