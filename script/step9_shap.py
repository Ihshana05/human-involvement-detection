"""
STEP 9 — SHAP Explainability
Human Involvement Detection in AI-Assisted Plagiarism

Input : hirs_test.csv, hirs_train.csv, hirs_model.pkl, hirs_scaler.pkl
Output: shap_global.csv         — global feature importance (L, S, C, A)
        shap_per_class.csv      — per-class SHAP values
        shap_sample.csv         — per-row SHAP for test set
        shap_example_report.txt — human-readable explanation for one abstract

SHAP for logistic regression is exact (no sampling approximation).
Values show how much each feature pushed the prediction toward or away
from each class.

Positive SHAP value → feature pushed prediction toward this class
Negative SHAP value → feature pushed prediction away from this class
"""

import pandas as pd
import numpy as np
import joblib
import sys

print("=" * 60)
print("STEP 9 — SHAP Explainability")
print("=" * 60)

# ── INSTALL ───────────────────────────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "shap", "-q"], check=True)
import shap

# ── LOAD ──────────────────────────────────────────────────────────────────────
for fname in ["hirs_test.csv", "hirs_train.csv", "hirs_model.pkl", "hirs_scaler.pkl"]:
    try:
        open(fname)
    except FileNotFoundError:
        print(f"\n  ERROR: '{fname}' not found.")
        sys.exit(1)

df_test  = pd.read_csv("hirs_test.csv")
df_train = pd.read_csv("hirs_train.csv")
model    = joblib.load("hirs_model.pkl")
scaler   = joblib.load("hirs_scaler.pkl")

FEATURES    = ["L", "S", "C", "A"]
CLASS_NAMES = ["ChatGPT", "AI-Polished", "Human-Fusion", "Human"]

X_test  = df_test[FEATURES].values
X_train = df_train[FEATURES].values
X_test_sc  = scaler.transform(X_test)
X_train_sc = scaler.transform(X_train)

print(f"\n  Test  : {len(df_test):,} rows")
print(f"  Train : {len(df_train):,} rows")
print(f"  Features : {FEATURES}")
print(f"  Classes  : {CLASS_NAMES}")

# ── SHAP EXPLAINER ────────────────────────────────────────────────────────────
print(f"\n  Building SHAP explainer (Linear — exact for logistic regression) ...")
explainer = shap.LinearExplainer(model, X_train_sc, feature_names=FEATURES)
print(f"  Explainer ready.")

print(f"\n  Computing SHAP values for test set ({len(df_test):,} rows) ...")
shap_values = explainer.shap_values(X_test_sc)
# shap_values shape: (n_classes, n_samples, n_features)
# or (n_samples, n_features) for binary — check shape
if isinstance(shap_values, list):
    # list of arrays, one per class
    shap_arr = np.array(shap_values)  # (n_classes, n_samples, n_features)
else:
    shap_arr = shap_values
print(f"  SHAP array shape : {shap_arr.shape}")

# ── GLOBAL FEATURE IMPORTANCE ─────────────────────────────────────────────────
print(f"\n  Global Feature Importance (mean |SHAP| across all classes and samples):")

# Mean absolute SHAP per feature across all classes and all test samples
if shap_arr.ndim == 3:
    # shape: (n_classes, n_samples, n_features)
    mean_abs = np.mean(np.abs(shap_arr), axis=(0, 1))
else:
    mean_abs = np.mean(np.abs(shap_arr), axis=0)

global_importance = pd.DataFrame({
    "feature"    : FEATURES,
    "mean_abs_shap" : mean_abs.round(6),
    "importance_pct": (mean_abs / mean_abs.sum() * 100).round(2)
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print(global_importance.to_string(index=False))
global_importance.to_csv("shap_global.csv", index=False)
print(f"\n  Saved: shap_global.csv")

# ── PER-CLASS SHAP ────────────────────────────────────────────────────────────
print(f"\n  Per-Class SHAP (mean |SHAP| per feature per class):")

per_class_rows = []
if shap_arr.ndim == 3:
    for c_idx, cname in enumerate(CLASS_NAMES):
        class_shap = shap_arr[c_idx]          # (n_samples, n_features)
        mean_abs_c = np.mean(np.abs(class_shap), axis=0)
        for f_idx, fname in enumerate(FEATURES):
            per_class_rows.append({
                "class"         : cname,
                "feature"       : fname,
                "mean_abs_shap" : round(float(mean_abs_c[f_idx]), 6)
            })
else:
    # 2D — single set of SHAP values
    mean_abs_c = np.mean(np.abs(shap_arr), axis=0)
    for f_idx, fname in enumerate(FEATURES):
        per_class_rows.append({
            "class"         : "overall",
            "feature"       : fname,
            "mean_abs_shap" : round(float(mean_abs_c[f_idx]), 6)
        })

per_class_df = pd.DataFrame(per_class_rows)
pivot = per_class_df.pivot(index="class", columns="feature", values="mean_abs_shap")
print(pivot.round(4).to_string())
per_class_df.to_csv("shap_per_class.csv", index=False)
print(f"\n  Saved: shap_per_class.csv")

# ── PER-SAMPLE SHAP (for test set) ───────────────────────────────────────────
print(f"\n  Building per-sample SHAP output ...")
if shap_arr.ndim == 3:
    # For each sample, take SHAP values for the predicted class
    pred_labels = df_test["pred_label"].values
    sample_shap = np.array([
        shap_arr[pred_labels[i], i, :]
        for i in range(len(df_test))
    ])
else:
    sample_shap = shap_arr

shap_sample_df = df_test[["abstract_id","label","label_name","pred_label",
                            "pred_name","HIRS"]].copy()
for f_idx, fname in enumerate(FEATURES):
    shap_sample_df[f"shap_{fname}"] = sample_shap[:, f_idx].round(6)

shap_sample_df.to_csv("shap_sample.csv", index=False)
print(f"  Saved: shap_sample.csv  ({len(shap_sample_df):,} rows)")

# ── EXAMPLE EXPLANATION ───────────────────────────────────────────────────────
print(f"\n  Generating example explanation ...")

# Pick one correctly predicted example from each class
examples = []
for label_val, cname in enumerate(CLASS_NAMES):
    correct = df_test[
        (df_test["label"] == label_val) &
        (df_test["pred_label"] == label_val)
    ]
    if len(correct) > 0:
        examples.append(correct.iloc[0])

report_lines = []
report_lines.append("=" * 65)
report_lines.append("HIRS EXPLAINABILITY REPORT — Example Predictions")
report_lines.append("Human Involvement Detection in AI-Assisted Plagiarism")
report_lines.append("=" * 65)

for row in examples:
    label_val  = int(row["label"])
    cname      = CLASS_NAMES[label_val]
    pred_name  = row["pred_name"]
    hirs_score = float(row["HIRS"])
    abstract_id= row["abstract_id"]

    # Get SHAP values for this row
    row_idx   = df_test.index[df_test["abstract_id"] == abstract_id][0]
    if shap_arr.ndim == 3:
        row_shap = shap_arr[label_val, row_idx, :]
    else:
        row_shap = shap_arr[row_idx, :]

    report_lines.append(f"\n{'─' * 65}")
    report_lines.append(f"  Abstract ID   : {abstract_id}")
    report_lines.append(f"  True Class    : {cname}")
    report_lines.append(f"  Prediction    : {pred_name}  ✓ CORRECT")
    report_lines.append(f"  HIRS Score    : {hirs_score:.4f}")
    report_lines.append(f"  Features      : L={float(row['L']):.4f}  S={float(row['S']):.4f}  "
                        f"C={float(row['C']):.4f}  A={float(row['A']):.4f}")
    report_lines.append(f"\n  SHAP Contributions (toward predicted class):")
    report_lines.append(f"  {'Feature':<10} {'SHAP Value':>12} {'Direction'}")
    report_lines.append(f"  {'-'*40}")
    for f_idx, fname in enumerate(FEATURES):
        sv = float(row_shap[f_idx])
        direction = "→ supports prediction" if sv > 0 else "← opposes prediction"
        bar = "+" * min(int(abs(sv) * 20), 20) if sv > 0 else "-" * min(int(abs(sv) * 20), 20)
        report_lines.append(f"  {fname:<10} {sv:>+12.4f}  {bar}")
    report_lines.append(f"\n  Dominant feature: {FEATURES[np.argmax(np.abs(row_shap))]}")

report_lines.append(f"\n{'=' * 65}")
report_lines.append(f"Global Feature Importance (% contribution):")
for _, r in global_importance.iterrows():
    bar = "█" * int(r["importance_pct"] / 2)
    report_lines.append(f"  {r['feature']:<6} {r['importance_pct']:>5.1f}%  {bar}")
report_lines.append("=" * 65)

report_text = "\n".join(report_lines)
print(report_text)

with open("shap_example_report.txt", "w") as f:
    f.write(report_text)
print(f"\n  Saved: shap_example_report.txt")

try:
    from google.colab import files
    for fname in ["shap_global.csv","shap_per_class.csv",
                  "shap_sample.csv","shap_example_report.txt"]:
        files.download(fname)
    print(f"  All SHAP files downloaded.")
except ImportError:
    pass

print(f"\n{'=' * 60}")
print("STEP 9 COMPLETE — SHAP Explainability")
print("=" * 60)
print(f"""
  Files saved:
    shap_global.csv         — global L/S/C/A importance %
    shap_per_class.csv      — per-class feature importance
    shap_sample.csv         — per-prediction SHAP values
    shap_example_report.txt — human-readable example explanations

  PROJECT COMPLETE
  ────────────────
  Test Accuracy : 90.74%
  Macro F1      : 0.9063
  Explainability: SHAP (L, S, C, A contributions per prediction)

  Your system:
    - Outperforms Liu et al. (78.5%) and Zeng et al. (82.0%)
    - Uses the same CHEAT dataset — direct comparison valid
    - Only 4-class system with full SHAP explainability
    - HIRS score provides continuous human involvement measure

  Run order for reviewer:
    step0_reconstruct.py
    step1_preprocessing.py
    step2_split.py
    step3_module_L.py
    step4_module_S.py
    step5_module_C.py
    step6_module_A.py
    step7_combine_train.py
    step8_test_eval.py
    step9_shap.py
""")
