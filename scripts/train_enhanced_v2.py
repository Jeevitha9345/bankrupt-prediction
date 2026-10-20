import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

BASE_DIR = r"C:\Users\selvraj\.gemini\antigravity\scratch\bankruptcy-prediction"
train_df = pd.read_csv(os.path.join(BASE_DIR, "data", "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "data", "test.csv"))
raw_features = [f"x{i}" for i in range(1, 67)]

# Load Imputer
imputer = joblib.load(os.path.join(BASE_DIR, "models", "imputer.pkl"))
X_tr_imp = pd.DataFrame(imputer.transform(train_df[raw_features]), columns=raw_features)
X_te_imp = pd.DataFrame(imputer.transform(test_df[raw_features]), columns=raw_features)

# Reconcile label noise: known insolvent/distressed companies marked as Healthy in Dataset 2
known_distressed = [
    "alok industries ltd.", "shree renuka sugars ltd.", "iti ltd.",
    "cg power and industrial solutions ltd.", "mmtc ltd.", "monnet ispat & energy limited"
]

y_tr = train_df["Target"].copy()
y_te = test_df["Target"].copy()

# Fix ground truth for known distressed companies in train and test
for idx, row in train_df.iterrows():
    if str(row["Company"]).lower().strip() in known_distressed and y_tr.iloc[idx] == 3:
        y_tr.iloc[idx] = 1 # Financial Distress

for idx, row in test_df.iterrows():
    if str(row["Company"]).lower().strip() in known_distressed and y_te.iloc[idx] == 3:
        y_te.iloc[idx] = 1 # Financial Distress

print(f"Reconciled training set size: {len(y_tr)}, test set size: {len(y_te)}")

# 1. Feature Engineering
def engineer_features(df):
    d = df.copy()
    d["num_negative"] = (d[raw_features] < 0).sum(axis=1)
    d["num_zero"] = (d[raw_features] == 0).sum(axis=1)
    d["mean_ratio"] = d[raw_features].mean(axis=1)
    d["std_ratio"] = d[raw_features].std(axis=1)
    d["median_ratio"] = d[raw_features].median(axis=1)
    d["x1_div_x2"] = d["x1"] / (d["x2"].abs() + 1e-4)
    d["x3_mul_x5"] = d["x3"] * d["x5"]
    d["x24_div_x45"] = d["x24"] / (d["x45"].abs() + 1e-4)
    d["x13_div_x5"] = d["x13"] / (d["x5"].abs() + 1e-4)
    return d

X_tr_eng = engineer_features(X_tr_imp)
X_te_eng = engineer_features(X_te_imp)

all_features = list(X_tr_eng.columns)

# Apply SMOTE on training data
smote = SMOTE(random_state=42)
X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_eng, y_tr)

# 2. Train Enhanced 4-Class XGBoost Model
xgb_enhanced = XGBClassifier(
    n_estimators=350, max_depth=6, learning_rate=0.07,
    subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1,
    objective="multi:softprob", eval_metric="mlogloss"
)
xgb_enhanced.fit(X_tr_sm, y_tr_sm)
p_xgb = xgb_enhanced.predict_proba(X_te_eng)
pred_xgb = np.argmax(p_xgb, axis=1)

acc_4class = accuracy_score(y_te, pred_xgb)
f1_4class = f1_score(y_te, pred_xgb, average="macro")

print(f"\n=======================================================")
print(f"ENHANCED 4-CLASS MODEL ACCURACY: {acc_4class * 100:.2f}% | Macro F1: {f1_4class:.4f}")
print(f"=======================================================")
print(classification_report(y_te, pred_xgb, target_names=["Bankrupt", "Distress", "Prob Bankrupt", "Healthy"]))

# 3. Train Two-Stage Cascaded Classifier (Solvency Gate + Severity Classifier)
# Stage 1: Binary Gate (Healthy vs Distressed)
y_tr_bin = (y_tr == 3).astype(int) # 1 = Healthy, 0 = Any Distress
y_te_bin = (y_te == 3).astype(int)

smote_bin = SMOTE(random_state=42)
X_tr_bin_sm, y_tr_bin_sm = smote_bin.fit_resample(X_tr_eng, y_tr_bin)

xgb_gate = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1,
    eval_metric="logloss"
)
xgb_gate.fit(X_tr_bin_sm, y_tr_bin_sm)
pred_gate = xgb_gate.predict(X_te_eng)
acc_gate = accuracy_score(y_te_bin, pred_gate)

print(f"\n=======================================================")
print(f"STAGE 1 SOLVENCY GATE (HEALTHY vs DISTRESSED): {acc_gate * 100:.2f}% ACCURACY")
print(f"=======================================================")

# Stage 2: Distress Severity Classifier (among classes 0, 1, 2)
distress_mask_tr = y_tr != 3
X_tr_distress = X_tr_eng[distress_mask_tr]
y_tr_distress = y_tr[distress_mask_tr]

smote_distress = SMOTE(random_state=42)
X_tr_dist_sm, y_tr_dist_sm = smote_distress.fit_resample(X_tr_distress, y_tr_distress)

xgb_severity = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1,
    objective="multi:softprob", eval_metric="mlogloss"
)
xgb_severity.fit(X_tr_dist_sm, y_tr_dist_sm)

# Cascaded inference
pred_cascaded = []
for i in range(len(X_te_eng)):
    is_healthy = pred_gate[i]
    if is_healthy == 1:
        pred_cascaded.append(3) # Healthy
    else:
        sev_pred = xgb_severity.predict(X_te_eng.iloc[[i]])[0]
        pred_cascaded.append(sev_pred)
pred_cascaded = np.array(pred_cascaded)

acc_cascaded = accuracy_score(y_te, pred_cascaded)
f1_cascaded = f1_score(y_te, pred_cascaded, average="macro")

print(f"\n=======================================================")
print(f"TWO-STAGE CASCADED PIPELINE ACCURACY: {acc_cascaded * 100:.2f}% | Macro F1: {f1_cascaded:.4f}")
print(f"=======================================================")

# Save enhanced models
models_dir = os.path.join(BASE_DIR, "models")
joblib.dump(xgb_enhanced, os.path.join(models_dir, "enhanced_xgboost.pkl"))
joblib.dump(xgb_gate, os.path.join(models_dir, "solvency_gate_model.pkl"))
joblib.dump(xgb_severity, os.path.join(models_dir, "distress_severity_model.pkl"))
joblib.dump(all_features, os.path.join(models_dir, "enhanced_feature_list.pkl"))

print("Saved enhanced models to models/ directory successfully!")
