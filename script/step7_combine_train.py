"""
STEP 7 — Combine + Train + HIRS Computation
Human Involvement Detection in AI-Assisted Plagiarism

Input : train_A.csv, val_A.csv, test_A.csv
        (these already contain L, S, C, A columns from previous steps)
Output: hirs_train.csv, hirs_val.csv, hirs_test.csv
        beta_coefficients.csv
        epoch_log.csv

Pipeline:
  1. Combine: use L, S, C, A columns already in the A files
  2. Scale: StandardScaler fitted on TRAIN only
  3. Train: Multinomial Logistic Regression on TRAIN
     X = [L, S, C, A]   y = label (0/1/2/3)
  4. 5-fold CV on TRAIN to verify genuine learning
  5. HIRS = 0.7 * P(Human) + 0.3 * max(P(Human-Fusion), P(Human))
  6. Evaluate on VAL (not test — test is saved for final evaluation)
  7. Save model + scaler for test evaluation

HIRS formula:
  Rewards texts with high human involvement at any level.
  P(Human) captures full human content.
  max term ensures partial involvement (Fusion) also scores high.
  Range: [0, 1]  — higher = more human involvement detected.

HIRS bands:
  0.00 - 0.25 : ChatGPT      (no human involvement)
  0.25 - 0.50 : AI-Polished  (low human involvement)
  0.50 - 0.75 : Human-Fusion (medium human involvement)
  0.75 - 1.00 : Human        (high human involvement)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
import joblib
import sys

print("=" * 60)
print("STEP 7 — Combine + Train + HIRS Computation")
print("=" * 60)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
for fname in ["train_A.csv", "val_A.csv", "test_A.csv"]:
    try:
        open(fname)
    except FileNotFoundError:
        print(f"\n  ERROR: '{fname}' not found.")
        sys.exit(1)

df_train = pd.read_csv("train_A.csv")
df_val   = pd.read_csv("val_A.csv")
df_test  = pd.read_csv("test_A.csv")

print(f"\n  Train : {len(df_train):,} | Val : {len(df_val):,} | Test : {len(df_test):,}")

# ── VERIFY FEATURES PRESENT ───────────────────────────────────────────────────
FEATURES = ["L", "S", "C", "A"]
for feat in FEATURES:
    if feat not in df_train.columns:
        print(f"\n  ERROR: Column '{feat}' not found in train_A.csv")
        print(f"  Available columns: {list(df_train.columns)}")
        sys.exit(1)

print(f"\n  Features confirmed: {FEATURES}")
print(f"\n  Feature means by class (TRAIN):")
means = df_train.groupby("label_name")[FEATURES].mean().round(4)
print(means.to_string())

# ── PREPARE ARRAYS ────────────────────────────────────────────────────────────
X_train = df_train[FEATURES].values
y_train = df_train["label"].values
X_val   = df_val[FEATURES].values
y_val   = df_val["label"].values
X_test  = df_test[FEATURES].values
y_test  = df_test["label"].values

# ── SCALE ─────────────────────────────────────────────────────────────────────
print(f"\n  Fitting StandardScaler on TRAIN only ...")
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)
print(f"  Scaler fitted. Feature means after scaling ≈ 0, std ≈ 1.")

# ── 5-FOLD CROSS VALIDATION ───────────────────────────────────────────────────
print(f"\n  5-Fold Stratified Cross-Validation on TRAIN ...")
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accs, cv_f1s = [], []

print(f"\n  {'Fold':<6} {'Train Acc':>10} {'Val Acc':>10} {'Val F1':>10}")
print(f"  {'-'*40}")

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_sc, y_train), 1):
    Xf_tr, Xf_va = X_train_sc[tr_idx], X_train_sc[va_idx]
    yf_tr, yf_va = y_train[tr_idx],    y_train[va_idx]

    clf = LogisticRegression(
        multi_class = "multinomial",
        solver      = "lbfgs",
        max_iter    = 2000,
        C           = 1.0,
        random_state= 42
    )
    clf.fit(Xf_tr, yf_tr)

    tr_acc = accuracy_score(yf_tr, clf.predict(Xf_tr)) * 100
    va_acc = accuracy_score(yf_va, clf.predict(Xf_va)) * 100
    va_f1  = f1_score(yf_va, clf.predict(Xf_va), average="macro")

    cv_accs.append(va_acc)
    cv_f1s.append(va_f1)
    print(f"  {fold:<6} {tr_acc:>9.2f}%  {va_acc:>9.2f}%  {va_f1:>10.4f}")

print(f"  {'-'*40}")
print(f"  {'Mean':<6} {'':>10} {np.mean(cv_accs):>9.2f}%  {np.mean(cv_f1s):>10.4f}")
print(f"  {'Std':<6} {'':>10} {np.std(cv_accs):>9.2f}%")

# ── TRAIN FINAL MODEL ON FULL TRAIN SET ───────────────────────────────────────
print(f"\n  Training final model on full TRAIN set ...")
model = LogisticRegression(
    multi_class = "multinomial",
    solver      = "lbfgs",
    max_iter    = 2000,
    C           = 1.0,
    random_state= 42
)
model.fit(X_train_sc, y_train)
print(f"  Converged: {model.n_iter_[0]} iterations")

# ── EVALUATE ON VAL ───────────────────────────────────────────────────────────
print(f"\n  Evaluating on VAL set ...")
y_val_pred = model.predict(X_val_sc)
train_acc  = accuracy_score(y_train, model.predict(X_train_sc)) * 100
val_acc    = accuracy_score(y_val,   y_val_pred) * 100
gap        = train_acc - val_acc

class_names = ["ChatGPT","AI-Polished","Human-Fusion","Human"]

print(f"\n  {'='*44}")
print(f"  Train Accuracy : {train_acc:.2f}%")
print(f"  Val   Accuracy : {val_acc:.2f}%")
print(f"  Gap            : {gap:.2f}%  ({'OK' if gap < 10 else 'Check for overfit'})")
print(f"  {'='*44}")

print(f"\n  Validation Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=class_names))

print(f"\n  Confusion Matrix (Val):")
cm = confusion_matrix(y_val, y_val_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print(cm_df.to_string())

# ── BETA COEFFICIENTS ─────────────────────────────────────────────────────────
print(f"\n  Learned Beta Coefficients (weights for L, S, C, A):")
beta_df = pd.DataFrame(
    model.coef_,
    columns = FEATURES,
    index   = [f"Class_{i}_{class_names[i]}" for i in range(4)]
)
print(beta_df.round(4).to_string())
beta_df.to_csv("beta_coefficients.csv")
print(f"\n  Saved: beta_coefficients.csv")

# ── HIRS COMPUTATION ──────────────────────────────────────────────────────────
print(f"\n  Computing HIRS scores ...")
print(f"  Formula: HIRS = 0.7 * P(Human) + 0.3 * max(P(Human-Fusion), P(Human))")

CLASS_ORDER = list(model.classes_)   # [0, 1, 2, 3]
IDX_HUMAN   = CLASS_ORDER.index(3)   # index for label=3 (Human)
IDX_FUSION  = CLASS_ORDER.index(2)   # index for label=2 (Human-Fusion)

def compute_hirs(X_sc, df_src, split_name):
    probs    = model.predict_proba(X_sc)
    p_human  = probs[:, IDX_HUMAN]
    p_fusion = probs[:, IDX_FUSION]
    hirs     = 0.7 * p_human + 0.3 * np.maximum(p_fusion, p_human)

    df_out = df_src.copy()
    df_out["P_ChatGPT"]     = probs[:, CLASS_ORDER.index(0)].round(6)
    df_out["P_AIPolished"]  = probs[:, CLASS_ORDER.index(1)].round(6)
    df_out["P_HumanFusion"] = probs[:, CLASS_ORDER.index(2)].round(6)
    df_out["P_Human"]       = p_human.round(6)
    df_out["HIRS"]          = hirs.round(6)
    df_out["pred_label"]    = model.predict(X_sc)
    df_out["pred_name"]     = [class_names[p] for p in model.predict(X_sc)]

    # HIRS band
    def hirs_band(h):
        if h < 0.25: return "ChatGPT (0.00-0.25)"
        if h < 0.50: return "AI-Polished (0.25-0.50)"
        if h < 0.75: return "Human-Fusion (0.50-0.75)"
        return "Human (0.75-1.00)"
    df_out["HIRS_band"] = df_out["HIRS"].apply(hirs_band)

    print(f"\n  HIRS distribution ({split_name}):")
    print(f"    Mean : {hirs.mean():.4f}")
    print(f"    Std  : {hirs.std():.4f}")
    print(f"    Min  : {hirs.min():.4f}")
    print(f"    Max  : {hirs.max():.4f}")

    print(f"\n  Mean HIRS per class ({split_name}):")
    means = df_out.groupby("label_name")["HIRS"].mean().round(4)
    print(means.to_string())

    print(f"\n  HIRS Band Analysis ({split_name}):")
    print(f"  {'Class':<16} {'Expected Band':<25} {'Mean HIRS':>10} {'In Band %':>10}")
    print(f"  {'-'*65}")
    bands = {
        "ChatGPT"      : (0.00, 0.25),
        "AI-Polished"  : (0.25, 0.50),
        "Human-Fusion" : (0.50, 0.75),
        "Human"        : (0.75, 1.00),
    }
    for lname, (lo, hi) in bands.items():
        cls_rows  = df_out[df_out["label_name"] == lname]
        mean_h    = cls_rows["HIRS"].mean()
        in_band   = ((cls_rows["HIRS"] >= lo) & (cls_rows["HIRS"] < hi)).mean() * 100
        print(f"  {lname:<16} {lo:.2f} – {hi:.2f}{'':>15} {mean_h:>10.4f} {in_band:>9.1f}%")

    return df_out

train_out = compute_hirs(X_train_sc, df_train, "TRAIN")
val_out   = compute_hirs(X_val_sc,   df_val,   "VAL")
test_out  = compute_hirs(X_test_sc,  df_test,  "TEST — save for final report")

# ── SAVE OUTPUTS ──────────────────────────────────────────────────────────────
print(f"\n  Saving ...")
train_out.to_csv("hirs_train.csv", index=False)
val_out.to_csv("hirs_val.csv",     index=False)
test_out.to_csv("hirs_test.csv",   index=False)
joblib.dump(model,  "hirs_model.pkl")
joblib.dump(scaler, "hirs_scaler.pkl")

# Save epoch/CV log
cv_log = pd.DataFrame({
    "fold"    : list(range(1, 6)),
    "val_acc" : cv_accs,
    "val_f1"  : cv_f1s
})
cv_log.to_csv("cv_log.csv", index=False)

print(f"    hirs_train.csv, hirs_val.csv, hirs_test.csv")
print(f"    beta_coefficients.csv")
print(f"    hirs_model.pkl, hirs_scaler.pkl")
print(f"    cv_log.csv")

try:
    from google.colab import files
    for fname in ["hirs_train.csv","hirs_val.csv","hirs_test.csv",
                  "beta_coefficients.csv","cv_log.csv"]:
        files.download(fname)
    print(f"  All files downloaded.")
except ImportError:
    pass

print(f"\n{'=' * 60}")
print("STEP 7 COMPLETE — Training + HIRS")
print("=" * 60)
print(f"""
  CV Mean Val Accuracy : {np.mean(cv_accs):.2f}% ± {np.std(cv_accs):.2f}%
  Train Accuracy       : {train_acc:.2f}%
  Val   Accuracy       : {val_acc:.2f}%
  Train-Val Gap        : {gap:.2f}%

  If Val Accuracy is 75-85% -> pipeline is working correctly.
  If Val Accuracy is below 70% -> share the beta coefficients
  and feature means — we will diagnose and fix.

  Next: Run step8_test_eval.py  (final test evaluation — once only)
  Do NOT retrain after seeing test results.
""")
