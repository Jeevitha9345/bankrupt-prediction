import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

BASE_DIR = r"C:\Users\selvraj\.gemini\antigravity\scratch\bankruptcy-prediction"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
FI_DIR = os.path.join(RESULTS_DIR, "feature_importance")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(FI_DIR, exist_ok=True)

# 1. Load the new Indian Company Financial Dataset
file_path = os.path.join(BASE_DIR, "Indian Company Financial Dataset.csv")
df = pd.read_csv(file_path)

print(f"Loaded Indian Company Financial Dataset: {len(df)} rows across {df['Company'].nunique()} companies.")

# 2. Derive 4 ground-truth classes using standard Indian banking / RBI / Altman financial rules
df["Net_Worth"] = df["Equity Share Capital"] + df["Reserves"]

def assign_4class(row):
    nw = row["Net_Worth"]
    ic = row["Interest_Coverage"]
    de = row["Debt_to_Equity"]
    np_val = row["Net profit"]
    om = row["Operating_Margin"]
    
    # Class 0: Bankrupt (Negative net worth or extreme default)
    if nw < 0 or (ic < -1.0 and np_val < -100):
        return 0
    # Class 2: Probably Bankrupt (Inability to cover interest + high leverage)
    elif ic < 1.0 or de > 4.0 or (np_val < 0 and ic < 1.5):
        return 2
    # Class 1: Financial Distress (Weak debt coverage or elevated leverage)
    elif ic < 2.5 or de > 2.0 or om < 0.02:
        return 1
    # Class 3: Healthy (Strong interest coverage, low/moderate debt, positive profits)
    else:
        return 3

df["Target"] = df.apply(assign_4class, axis=1)
class_names = ["Bankrupt", "Financial Distress", "Probably Bankrupt", "Healthy"]

print("\n--- 4-CLASS DISTRIBUTION ---")
for tid, tname in enumerate(class_names):
    cnt = (df["Target"] == tid).sum()
    print(f"  {tname} ({tid}): {cnt} ({cnt/len(df)*100:.2f}%)")

# Feature columns (exclude identifiers & helper columns)
feature_cols = [c for c in df.columns if c not in ["Company", "Year", "Target", "Net_Worth"]]
print(f"Total financial features: {len(feature_cols)}")

X = df[feature_cols].copy()
y = df["Target"].copy()
groups = df["Company"]

# Impute median on features
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
joblib.dump(imputer, os.path.join(MODELS_DIR, "indian_imputer.pkl"))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, "indian_feature_list.pkl"))

# Scaler for Logistic Regression
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=feature_cols)
joblib.dump(scaler, os.path.join(MODELS_DIR, "indian_scaler.pkl"))

# 3. Company-Level Split (Zero leakage between train and test)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X_imp, y, groups))

X_tr = X_imp.iloc[train_idx].copy().reset_index(drop=True)
y_tr = y.iloc[train_idx].copy().reset_index(drop=True)
X_te = X_imp.iloc[test_idx].copy().reset_index(drop=True)
y_te = y.iloc[test_idx].copy().reset_index(drop=True)

X_tr_sc = X_scaled.iloc[train_idx].copy().reset_index(drop=True)
X_te_sc = X_scaled.iloc[test_idx].copy().reset_index(drop=True)

print(f"\nTraining set: {len(X_tr)} rows ({groups.iloc[train_idx].nunique()} companies)")
print(f"Testing set:  {len(X_te)} rows ({groups.iloc[test_idx].nunique()} companies)")

# 4. Train the 5 algorithms
models = {
    "XGBoost": (
        XGBClassifier(n_estimators=350, max_depth=6, learning_rate=0.08, subsample=0.85,
                      colsample_bytree=0.85, random_state=42, n_jobs=-1,
                      objective="multi:softprob", eval_metric="mlogloss"),
        X_tr, X_te, "xgboost_indian.pkl"
    ),
    "LightGBM": (
        LGBMClassifier(n_estimators=350, num_leaves=45, learning_rate=0.08, max_depth=10,
                       random_state=42, verbose=-1, n_jobs=-1),
        X_tr, X_te, "lightgbm_indian.pkl"
    ),
    "CatBoost": (
        CatBoostClassifier(iterations=350, depth=6, learning_rate=0.08, random_seed=42,
                           verbose=0, thread_count=-1),
        X_tr, X_te, "catboost_indian.pkl"
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=250, max_depth=20, random_state=42, n_jobs=-1),
        X_tr, X_te, "random_forest_indian.pkl"
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        X_tr_sc, X_te_sc, "logistic_regression_indian.pkl"
    )
}

results_table = []

print("\n--- TRAINING AND EVALUATING ALL 5 MODELS ---")
for model_name, (clf, tr_data, te_data, filename) in models.items():
    clf.fit(tr_data, y_tr)
    preds = clf.predict(te_data)
    if hasattr(preds, "flatten"):
        preds = preds.flatten()
        
    acc = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, average="macro", zero_division=0)
    rec = recall_score(y_te, preds, average="macro", zero_division=0)
    f1_macro = f1_score(y_te, preds, average="macro", zero_division=0)
    f1_wt = f1_score(y_te, preds, average="weighted", zero_division=0)
    
    # Per-class recall
    rec_per_class = recall_score(y_te, preds, average=None, zero_division=0)
    
    # Save model
    joblib.dump(clf, os.path.join(MODELS_DIR, filename))
    
    results_table.append({
        "Model": model_name,
        "Accuracy": f"{acc * 100:.2f}%",
        "Macro F1": round(f1_macro, 4),
        "Weighted F1": round(f1_wt, 4),
        "Bankrupt Recall": f"{rec_per_class[0]*100:.1f}%",
        "Financial Distress Recall": f"{rec_per_class[1]*100:.1f}%",
        "Probably Bankrupt Recall": f"{rec_per_class[2]*100:.1f}%",
        "Healthy Recall": f"{rec_per_class[3]*100:.1f}%"
    })
    
    # Confusion Matrix
    cm = confusion_matrix(y_te, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {model_name} (Acc: {acc*100:.2f}%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(CM_DIR, f"cm_indian_{model_name.lower().replace(' ', '_')}.png"), dpi=300)
    plt.close()

res_df = pd.DataFrame(results_table)
res_df.to_csv(os.path.join(RESULTS_DIR, "indian_dataset_model_comparison.csv"), index=False)

print("\n================================================================================")
print("FINAL 4-CLASS COMPARISON TABLE (OUT-OF-SAMPLE UNSEEN INDIAN COMPANIES)")
print("================================================================================")
print(res_df.to_string(index=False))
print("================================================================================")
