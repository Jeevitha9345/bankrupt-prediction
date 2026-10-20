import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

BASE_DIR = r"C:\Users\selvraj\.gemini\antigravity\scratch\bankruptcy-prediction"
train_df = pd.read_csv(os.path.join(BASE_DIR, "data", "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "data", "test.csv"))

raw_feats = [f"x{i}" for i in range(1, 67)]

# Load Imputer
imputer = joblib.load(os.path.join(BASE_DIR, "models", "imputer.pkl"))
X_tr_imp = pd.DataFrame(imputer.transform(train_df[raw_feats]), columns=raw_feats)
X_te_imp = pd.DataFrame(imputer.transform(test_df[raw_feats]), columns=raw_feats)

def add_features(df_imp):
    df = df_imp.copy()
    df["num_negative"] = (df[raw_feats] < 0).sum(axis=1)
    df["num_zero"] = (df[raw_feats] == 0).sum(axis=1)
    df["mean_ratio"] = df[raw_feats].mean(axis=1)
    df["std_ratio"] = df[raw_feats].std(axis=1)
    df["median_ratio"] = df[raw_feats].median(axis=1)
    
    # Financial interaction proxies
    df["x1_div_x2"] = df["x1"] / (df["x2"].abs() + 1e-4)
    df["x3_mul_x5"] = df["x3"] * df["x5"]
    df["x24_div_x45"] = df["x24"] / (df["x45"].abs() + 1e-4)
    df["x13_div_x5"] = df["x13"] / (df["x5"].abs() + 1e-4)
    df["x8_mul_x16"] = df["x8"] * df["x16"]
    return df

X_tr_eng = add_features(X_tr_imp)
X_te_eng = add_features(X_te_imp)

y_tr = train_df["Target"].values
y_te = test_df["Target"].values

print(f"Features: {len(X_tr_eng.columns)}")

smote = SMOTE(random_state=42)
X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_eng, y_tr)

# 1. XGBoost
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1,
    objective="multi:softprob", eval_metric="mlogloss"
)
xgb.fit(X_tr_sm, y_tr_sm)
p_xgb = xgb.predict_proba(X_te_eng)
pred_xgb = np.argmax(p_xgb, axis=1)
print(f"XGBoost Test Accuracy:  {accuracy_score(y_te, pred_xgb):.4f} | Macro F1: {f1_score(y_te, pred_xgb, average='macro'):.4f}")

# 2. LightGBM
lgb = LGBMClassifier(
    n_estimators=300, num_leaves=45, learning_rate=0.08,
    max_depth=10, random_state=42, verbose=-1, n_jobs=-1
)
lgb.fit(X_tr_sm, y_tr_sm)
p_lgb = lgb.predict_proba(X_te_eng)
pred_lgb = np.argmax(p_lgb, axis=1)
print(f"LightGBM Test Accuracy: {accuracy_score(y_te, pred_lgb):.4f} | Macro F1: {f1_score(y_te, pred_lgb, average='macro'):.4f}")

# 3. CatBoost
cb = CatBoostClassifier(
    iterations=350, depth=6, learning_rate=0.08,
    random_seed=42, verbose=0, thread_count=-1
)
cb.fit(X_tr_sm, y_tr_sm)
p_cb = cb.predict_proba(X_te_eng)
pred_cb = np.argmax(p_cb, axis=1)
print(f"CatBoost Test Accuracy: {accuracy_score(y_te, pred_cb):.4f} | Macro F1: {f1_score(y_te, pred_cb, average='macro'):.4f}")

# 4. ExtraTrees
et = ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
et.fit(X_tr_sm, y_tr_sm)
p_et = et.predict_proba(X_te_eng)
pred_et = np.argmax(p_et, axis=1)
print(f"ExtraTrees Test Accuracy: {accuracy_score(y_te, pred_et):.4f} | Macro F1: {f1_score(y_te, pred_et, average='macro'):.4f}")

# Blended ensemble
p_blend = 0.35 * p_xgb + 0.30 * p_lgb + 0.20 * p_cb + 0.15 * p_et
pred_blend = np.argmax(p_blend, axis=1)
print(f"Ensemble Blend Accuracy:  {accuracy_score(y_te, pred_blend):.4f} | Macro F1: {f1_score(y_te, pred_blend, average='macro'):.4f}")
