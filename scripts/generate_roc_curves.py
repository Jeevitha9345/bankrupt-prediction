import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

base_dir = r"C:\Users\selvraj\.gemini\antigravity\scratch\bankruptcy-prediction"
results_dir = os.path.join(base_dir, "results")
artifacts_dir = r"C:\Users\selvraj\.gemini\antigravity\brain\2a30a551-d275-46ad-af8f-06f46c7289ac"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)

# 1. Load data & assign ground truth classes
df = pd.read_csv(os.path.join(base_dir, "Indian Company Financial Dataset.csv"))
df["Net_Worth"] = df["Equity Share Capital"] + df["Reserves"]

def assign_4class(row):
    nw = row["Net_Worth"]
    ic = row["Interest_Coverage"]
    de = row["Debt_to_Equity"]
    np_val = row["Net profit"]
    om = row["Operating_Margin"]
    if nw < 0 or (ic < -1.0 and np_val < -100):
        return 0
    elif ic < 1.0 or de > 4.0 or (np_val < 0 and ic < 1.5):
        return 2
    elif ic < 2.5 or de > 2.0 or om < 0.02:
        return 1
    else:
        return 3

df["Target"] = df.apply(assign_4class, axis=1)
feature_cols = [c for c in df.columns if c not in ["Company", "Year", "Target", "Net_Worth"]]

imputer = joblib.load(os.path.join(base_dir, "models", "indian_imputer.pkl"))
scaler = joblib.load(os.path.join(base_dir, "models", "indian_scaler.pkl"))

X = df[feature_cols].copy()
y = df["Target"].copy()
groups = df["Company"]

X_imp = pd.DataFrame(imputer.transform(X), columns=feature_cols)
X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=feature_cols)

# 2. Company-level test split (201 unseen test companies)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X_imp, y, groups))

X_te = X_imp.iloc[test_idx].copy().reset_index(drop=True)
y_te = y.iloc[test_idx].copy().reset_index(drop=True)
X_te_sc = X_scaled.iloc[test_idx].copy().reset_index(drop=True)

y_te_bin = label_binarize(y_te, classes=[0, 1, 2, 3])
n_classes = 4
class_names = ["Bankrupt", "Financial Distress", "Probably Bankrupt", "Healthy"]

models = {
    "CatBoost": (joblib.load(os.path.join(base_dir, "models", "catboost_indian.pkl")), X_te, "#1E3A8A", "-", 2.5),
    "LightGBM": (joblib.load(os.path.join(base_dir, "models", "lightgbm_indian.pkl")), X_te, "#059669", "--", 2.0),
    "XGBoost": (joblib.load(os.path.join(base_dir, "models", "xgboost_indian.pkl")), X_te, "#D97706", "-.", 2.0),
    "Random Forest": (joblib.load(os.path.join(base_dir, "models", "random_forest_indian.pkl")), X_te, "#7C3AED", ":", 2.0),
    "Logistic Regression": (joblib.load(os.path.join(base_dir, "models", "logistic_regression_indian.pkl")), X_te_sc, "#DC2626", "-", 1.8)
}

# Compute ROC curves for each model
model_roc_data = {}
for name, (clf, data, color, lstyle, lwidth) in models.items():
    probs = clf.predict_proba(data)
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_te_bin[:, i], probs[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        
    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    
    model_roc_data[name] = {
        "all_fpr": all_fpr,
        "mean_tpr": mean_tpr,
        "macro_auc": macro_auc,
        "color": color,
        "lstyle": lstyle,
        "lwidth": lwidth,
        "fpr_dict": fpr_dict,
        "tpr_dict": tpr_dict,
        "roc_auc_dict": roc_auc_dict
    }

# =========================================================================
# FIGURE 6 (PRIMARY): Comparative ROC Curves of All Investigated Models
# =========================================================================
plt.figure(figsize=(9, 7), dpi=300)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

for name, mdata in model_roc_data.items():
    plt.plot(
        mdata["all_fpr"], mdata["mean_tpr"],
        label=f"{name} (AUC = {mdata['macro_auc']:.4f})",
        color=mdata["color"],
        linestyle=mdata["lstyle"],
        linewidth=mdata["lwidth"]
    )

# Random guess diagonal
plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier (AUC = 0.5000)", alpha=0.7)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=13, fontweight="bold", labelpad=10)
plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=13, fontweight="bold", labelpad=10)
plt.title("Figure 6: ROC Curves of the Investigated Models\n(Evaluated on 1,839 Unseen Indian Corporate Filings)", fontsize=14, fontweight="bold", pad=15)
plt.legend(loc="lower right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=11, framealpha=0.95)
plt.grid(True, linestyle=":", alpha=0.6, color="#94A3B8")
plt.tight_layout()

# Save primary Figure 6
fig6_path = os.path.join(results_dir, "figure_6_roc_curves.png")
fig6_artifact = os.path.join(artifacts_dir, "figure_6_roc_curves.png")
plt.savefig(fig6_path, dpi=300, bbox_inches="tight")
plt.savefig(fig6_artifact, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Figure 6 to: {fig6_path}")

# =========================================================================
# FIGURE 6 (DETAILED 2-PANEL): Model Comparison + CatBoost Class-Wise ROC
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Panel A: All models comparison
for name, mdata in model_roc_data.items():
    ax1.plot(
        mdata["all_fpr"], mdata["mean_tpr"],
        label=f"{name} (AUC = {mdata['macro_auc']:.4f})",
        color=mdata["color"],
        linestyle=mdata["lstyle"],
        linewidth=mdata["lwidth"]
    )
ax1.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Chance (AUC = 0.5000)", alpha=0.7)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.05])
ax1.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
ax1.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
ax1.set_title("(A) Macro-Average ROC Curves Across Models", fontsize=13, fontweight="bold")
ax1.legend(loc="lower right", frameon=True, fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)

# Panel B: Class-wise ROC for best model (CatBoost)
cb_data = model_roc_data["CatBoost"]
class_colors = ["#EF4444", "#F59E0B", "#F97316", "#10B981"]
for i in range(n_classes):
    ax2.plot(
        cb_data["fpr_dict"][i], cb_data["tpr_dict"][i],
        label=f"Class {i}: {class_names[i]} (AUC = {cb_data['roc_auc_dict'][i]:.4f})",
        color=class_colors[i],
        linewidth=2.2
    )
ax2.plot(
    cb_data["all_fpr"], cb_data["mean_tpr"],
    label=f"Macro-Average (AUC = {cb_data['macro_auc']:.4f})",
    color="#1E3A8A", linestyle="--", linewidth=2.5
)
ax2.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Chance (AUC = 0.5000)", alpha=0.7)
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.05])
ax2.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
ax2.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
ax2.set_title("(B) CatBoost Per-Class ROC Analysis (Best Model)", fontsize=13, fontweight="bold")
ax2.legend(loc="lower right", frameon=True, fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)

plt.suptitle("Figure 6: Comprehensive ROC and AUC Analysis for Corporate Bankruptcy Prediction", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()

detailed_fig_path = os.path.join(results_dir, "figure_6_roc_analysis_detailed.png")
detailed_artifact = os.path.join(artifacts_dir, "figure_6_roc_analysis_detailed.png")
plt.savefig(detailed_fig_path, dpi=300, bbox_inches="tight")
plt.savefig(detailed_artifact, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Detailed Figure 6 to: {detailed_fig_path}")
