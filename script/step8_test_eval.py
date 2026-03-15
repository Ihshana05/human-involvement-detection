"""
STEP 8 — Final Test Evaluation
Human Involvement Detection in AI-Assisted Plagiarism

Input : hirs_test.csv  (already has HIRS, pred_label, pred_name)
        OR test_A.csv + hirs_model.pkl + hirs_scaler.pkl

Output: test_evaluation_report.txt
        confusion_matrix.csv
        classification_report.csv

WARNING: Run this ONCE only. Do not retrain after seeing results.
The test set must remain completely unseen during all development.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              precision_score, recall_score)
import sys

print("=" * 60)
print("STEP 8 — Final Test Evaluation")
print("⚠️  RUN ONCE ONLY — DO NOT RETRAIN AFTER THIS")
print("=" * 60)

# ── LOAD TEST RESULTS ─────────────────────────────────────────────────────────
# hirs_test.csv already has predictions from Step 7
try:
    df = pd.read_csv("hirs_test.csv")
except FileNotFoundError:
    print("\n  ERROR: 'hirs_test.csv' not found.")
    print("  Upload hirs_test.csv (downloaded from Step 7).")
    sys.exit(1)

print(f"\n  Loaded : {len(df):,} rows")
print(f"  Columns: {list(df.columns)}")

y_true = df["label"].values
y_pred = df["pred_label"].values
class_names = ["ChatGPT", "AI-Polished", "Human-Fusion", "Human"]

# ── CORE METRICS ──────────────────────────────────────────────────────────────
acc       = accuracy_score(y_true, y_pred) * 100
macro_f1  = f1_score(y_true, y_pred, average="macro")
macro_p   = precision_score(y_true, y_pred, average="macro")
macro_r   = recall_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")

print(f"\n{'=' * 50}")
print(f"  FINAL TEST RESULTS")
print(f"{'=' * 50}")
print(f"  Test Accuracy      : {acc:.2f}%")
print(f"  Macro F1           : {macro_f1:.4f}")
print(f"  Macro Precision    : {macro_p:.4f}")
print(f"  Macro Recall       : {macro_r:.4f}")
print(f"  Weighted F1        : {weighted_f1:.4f}")
print(f"{'=' * 50}")

# ── PER CLASS ─────────────────────────────────────────────────────────────────
print(f"\n  Per-Class Performance:")
report = classification_report(
    y_true, y_pred,
    target_names = class_names,
    output_dict  = True
)
report_str = classification_report(y_true, y_pred, target_names=class_names)
print(report_str)

# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────
cm     = confusion_matrix(y_true, y_pred)
cm_df  = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
cm_norm_df = pd.DataFrame(
    cm_norm.round(3),
    index   = class_names,
    columns = class_names
)

print(f"  Confusion Matrix (counts):")
print(cm_df.to_string())
print(f"\n  Confusion Matrix (normalised by true class):")
print(cm_norm_df.to_string())

# ── HIRS ANALYSIS ─────────────────────────────────────────────────────────────
print(f"\n  HIRS Score Analysis:")
print(f"  {'Class':<16} {'Mean HIRS':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
print(f"  {'-'*55}")
for lname in class_names:
    cls_h = df[df["label_name"] == lname]["HIRS"]
    print(f"  {lname:<16} {cls_h.mean():>10.4f} {cls_h.std():>8.4f} "
          f"{cls_h.min():>8.4f} {cls_h.max():>8.4f}")

# ── COMPARISON WITH EXISTING SYSTEMS ─────────────────────────────────────────
print(f"\n  Comparison with Existing Systems (same CHEAT dataset):")
print(f"  {'System':<25} {'Dataset':<8} {'Classes':>7} {'Accuracy':>10} {'Explainable':>12}")
print(f"  {'-'*65}")
systems = [
    ("Liu et al. (2023)",    "CHEAT", 4, "78.5%",  "No"),
    ("Zeng et al. (2024)",   "CHEAT", 4, "82.0%",  "No"),
    ("GPTZero (binary)",     "Mixed", 2, "~99%*",  "No"),
    ("GLTR (binary)",        "Mixed", 2, "~80%*",  "Partial"),
    (f"HIRS (This Work)",    "CHEAT", 4, f"{acc:.1f}%", "Yes (SHAP)"),
]
for name, ds, cls, a, exp in systems:
    marker = " ← OUR SYSTEM" if "This Work" in name else ""
    print(f"  {name:<25} {ds:<8} {cls:>7} {a:>10} {exp:>12}{marker}")
print(f"\n  (*) Binary classification — not directly comparable")

# ── SAVE REPORTS ──────────────────────────────────────────────────────────────
# Classification report CSV
report_rows = []
for cls_name in class_names:
    r = report[cls_name]
    report_rows.append({
        "class"     : cls_name,
        "precision" : round(r["precision"], 4),
        "recall"    : round(r["recall"],    4),
        "f1_score"  : round(r["f1-score"],  4),
        "support"   : int(r["support"])
    })
report_df = pd.DataFrame(report_rows)
report_df.to_csv("classification_report.csv", index=False)

cm_df.to_csv("confusion_matrix.csv")
cm_norm_df.to_csv("confusion_matrix_normalised.csv")

# Full text report
with open("test_evaluation_report.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("HIRS — Final Test Evaluation Report\n")
    f.write("Human Involvement Detection in AI-Assisted Plagiarism\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Test Accuracy      : {acc:.2f}%\n")
    f.write(f"Macro F1           : {macro_f1:.4f}\n")
    f.write(f"Macro Precision    : {macro_p:.4f}\n")
    f.write(f"Macro Recall       : {macro_r:.4f}\n")
    f.write(f"Weighted F1        : {weighted_f1:.4f}\n\n")
    f.write("Per-Class Report:\n")
    f.write(report_str + "\n")
    f.write("Confusion Matrix:\n")
    f.write(cm_df.to_string() + "\n\n")
    f.write("Normalised Confusion Matrix:\n")
    f.write(cm_norm_df.to_string() + "\n")

print(f"\n  Saved:")
print(f"    test_evaluation_report.txt")
print(f"    classification_report.csv")
print(f"    confusion_matrix.csv")
print(f"    confusion_matrix_normalised.csv")

try:
    from google.colab import files
    for fname in ["test_evaluation_report.txt",
                  "classification_report.csv",
                  "confusion_matrix.csv",
                  "confusion_matrix_normalised.csv"]:
        files.download(fname)
    print(f"  All files downloaded.")
except ImportError:
    pass

print(f"\n{'=' * 60}")
print("STEP 8 COMPLETE — Final Test Evaluation")
print("=" * 60)
print(f"""
  Test Accuracy : {acc:.2f}%
  Macro F1      : {macro_f1:.4f}

  These are your FINAL numbers — do not retrain.

  Next: Run step9_shap.py  (SHAP explainability)
  SHAP explains which feature (L, S, C, A) contributed most
  to each prediction — this is your system's key advantage
  over all existing tools.
""")
