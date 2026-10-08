import pandas as pd
import numpy as np
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR = r"C:\Users\selvraj\.gemini\antigravity\scratch\bankruptcy-prediction"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

features = [f"x{i}" for i in range(1, 67)]
X_train_raw = train_df[features].copy()
y_train = train_df["Target"].copy()
X_test_raw = test_df[features].copy()
y_test = test_df["Target"].copy()

print(f"X_train_raw shape: {X_train_raw.shape}")
print(f"X_test_raw shape: {X_test_raw.shape}")

# 1. Fit Imputer strictly on Training data
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train_raw)

# Save imputer
joblib.dump(imputer, os.path.join(MODELS_DIR, "imputer.pkl"))
print("Fitted and saved models/imputer.pkl (median strategy on training data)")

# Transform Train and Test
X_train_imputed = pd.DataFrame(imputer.transform(X_train_raw), columns=features)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_raw), columns=features)

# 2. Fit Scaler strictly on Imputed Training data
scaler = StandardScaler()
scaler.fit(X_train_imputed)

# Save scaler
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print("Fitted and saved models/scaler.pkl (StandardScaler on training data)")

# 3. Save feature list and label mapping
joblib.dump(features, os.path.join(MODELS_DIR, "feature_list.pkl"))
print("Saved models/feature_list.pkl")

label_mapping = {
    0: "Bankrupt",
    1: "Financial Distress",
    2: "Probably Bankrupt",
    3: "Healthy"
}
joblib.dump(label_mapping, os.path.join(MODELS_DIR, "label_encoder.pkl"))
print("Saved models/label_encoder.pkl")

# 4. Correlation Analysis on Training data
print("\n--- CORRELATION ANALYSIS (|r| >= 0.95) ---")
corr_matrix = X_train_imputed.corr(method="pearson")

high_corr_pairs = []
seen_pairs = set()

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        f1 = features[i]
        f2 = features[j]
        val = corr_matrix.loc[f1, f2]
        if abs(val) >= 0.95:
            high_corr_pairs.append({
                "Feature_1": f1,
                "Feature_2": f2,
                "Correlation": round(float(val), 4),
                "Abs_Correlation": round(float(abs(val)), 4)
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(by="Abs_Correlation", ascending=False)
corr_report_path = os.path.join(RESULTS_DIR, "feature_correlations_above_95.csv")
high_corr_df.to_csv(corr_report_path, index=False)
print(f"Total feature pairs with |r| >= 0.95: {len(high_corr_df)}")
print(high_corr_df.to_string(index=False))

# Document feature selection rationale:
# Tree models (RF, XGBoost, CatBoost, LightGBM) natively handle multicollinearity via recursive partitioning / regularization.
# For Logistic Regression, L2 regularization mitigates multicollinearity without needing aggressive heuristic dropping.
# We retain all 66 features in the benchmark dataset to preserve complete financial signal, while saving the correlation report.
