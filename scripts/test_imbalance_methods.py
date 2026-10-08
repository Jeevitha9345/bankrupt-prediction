import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

BASE_DIR = r"C:\Users\selvraj\.gemini\antigravity\scratch\bankruptcy-prediction"
train_df = pd.read_csv(os.path.join(BASE_DIR, "data", "train.csv"))

features = [f"x{i}" for i in range(1, 67)]
X_train = train_df[features]
y_train = train_df["Target"]
groups = train_df["Company"]

imputer = joblib.load(os.path.join(BASE_DIR, "models", "imputer.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

X_train_imp = pd.DataFrame(imputer.transform(X_train), columns=features)
X_train_scaled = pd.DataFrame(scaler.transform(X_train_imp), columns=features)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model_type in [("Logistic Regression", "lr"), ("Random Forest", "rf")]:
    f1_weights_list = []
    f1_smote_list = []
    
    for tr_idx, val_idx in sgkf.split(X_train, y_train, groups):
        # Validation fold
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        if model_type == "lr":
            X_tr, X_val = X_train_scaled.iloc[tr_idx], X_train_scaled.iloc[val_idx]
            # Approach A: Class Weights
            clf_cw = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
            clf_cw.fit(X_tr, y_tr)
            pred_cw = clf_cw.predict(X_val)
            f1_weights_list.append(f1_score(y_val, pred_cw, average="macro"))
            
            # Approach B: SMOTE on training fold only
            smote = SMOTE(random_state=42)
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
            clf_sm = LogisticRegression(max_iter=2000, random_state=42)
            clf_sm.fit(X_tr_sm, y_tr_sm)
            pred_sm = clf_sm.predict(X_val)
            f1_smote_list.append(f1_score(y_val, pred_sm, average="macro"))
            
        elif model_type == "rf":
            X_tr, X_val = X_train_imp.iloc[tr_idx], X_train_imp.iloc[val_idx]
            # Approach A: Class Weights
            clf_cw = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
            clf_cw.fit(X_tr, y_tr)
            pred_cw = clf_cw.predict(X_val)
            f1_weights_list.append(f1_score(y_val, pred_cw, average="macro"))
            
            # Approach B: SMOTE
            smote = SMOTE(random_state=42)
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
            clf_sm = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf_sm.fit(X_tr_sm, y_tr_sm)
            pred_sm = clf_sm.predict(X_val)
            f1_smote_list.append(f1_score(y_val, pred_sm, average="macro"))
            
    print(f"{model_name} CV Macro F1:")
    print(f"  Class Weights: {np.mean(f1_weights_list):.4f} +/- {np.std(f1_weights_list):.4f}")
    print(f"  SMOTE:         {np.mean(f1_smote_list):.4f} +/- {np.std(f1_smote_list):.4f}")
