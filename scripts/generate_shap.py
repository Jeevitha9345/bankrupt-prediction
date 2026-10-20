import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\selvraj\.gemini\antigravity\scratch\bankruptcy-prediction"
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "feature_importance")

features = joblib.load(os.path.join(MODELS_DIR, "indian_feature_list.pkl"))
imputer = joblib.load(os.path.join(MODELS_DIR, "indian_imputer.pkl"))
model = joblib.load(os.path.join(MODELS_DIR, "catboost_indian.pkl"))
df = pd.read_csv(os.path.join(BASE_DIR, "Indian Company Financial Dataset.csv"))

sample_df = df[features].sample(500, random_state=42)
sample_imp = pd.DataFrame(imputer.transform(sample_df), columns=features)

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(sample_imp)
print("SHAP values shape:", shap_vals.shape)

class_names = ["Bankrupt", "Financial Distress", "Probably Bankrupt", "Healthy"]
for cid, cname in enumerate(class_names):
    fname = cname.lower().replace(" ", "_")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals[:, :, cid], sample_imp, show=False, max_display=12)
    plt.title(f"SHAP Explanations - {cname}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"shap_{fname}.png"), dpi=300)
    plt.close()
    print(f"Saved SHAP plot for {cname}")

# Compute mean absolute SHAP values for global ranking
mean_abs_shap = np.mean(np.abs(shap_vals), axis=(0, 2))
shap_ranking = pd.DataFrame({
    "Feature": features,
    "Mean_Abs_SHAP": mean_abs_shap
}).sort_values(by="Mean_Abs_SHAP", ascending=False)

shap_ranking.head(20).to_csv(os.path.join(RESULTS_DIR, "shap_feature_importance.csv"), index=False)
print("Saved results/feature_importance/shap_feature_importance.csv")
