import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = r"C:\Users\selvraj\.gemini\antigravity\scratch\bankruptcy-prediction"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("--- STEP 1: LOAD AND COMBINE RAW DATASETS ---")
df1 = pd.read_csv(os.path.join(DATA_DIR, "dataset1_raw.csv"))
df2 = pd.read_csv(os.path.join(DATA_DIR, "dataset2_raw.csv"))

print(f"Dataset 1 rows: {len(df1)}, columns: {len(df1.columns)}")
print(f"Dataset 2 rows: {len(df2)}, columns: {len(df2.columns)}")

df1.columns = [c.strip() for c in df1.columns]
df2.columns = [c.strip() for c in df2.columns]

# Create combined_raw.csv
combined_raw = pd.concat([df1, df2], ignore_index=True)
combined_raw_path = os.path.join(DATA_DIR, "combined_raw.csv")
combined_raw.to_csv(combined_raw_path, index=False)
print(f"Saved combined_raw.csv: {len(combined_raw)} rows, {len(combined_raw.columns)} columns")

label_map_standard = {
    "bankrupt": "Bankrupt",
    "probable bankrupt": "Probably Bankrupt",
    "probably bankrupt": "Probably Bankrupt",
    "financial distress": "Financial Distress",
    "healthy": "Healthy"
}

label_to_numeric = {
    "Bankrupt": 0,
    "Financial Distress": 1,
    "Probably Bankrupt": 2,
    "Healthy": 3
}

combined_raw["Standard_Label"] = (
    combined_raw["Label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(label_map_standard)
)

print("\n--- COMBINED RAW CLASS DISTRIBUTION ---")
print(combined_raw["Standard_Label"].value_counts(dropna=False))
print(f"Unlabeled / NaN rows in combined_raw: {combined_raw['Standard_Label'].isna().sum()}")

# Cleaned dataset: Exclude rows with missing labels
cleaned_df = combined_raw[combined_raw["Standard_Label"].notna()].copy()
print(f"\nLabeled rows for modeling: {len(cleaned_df)}")

cleaned_df["Target"] = cleaned_df["Standard_Label"].map(label_to_numeric)
cleaned_df["Company"] = cleaned_df["Folder_name"].astype(str).str.strip()
cleaned_df["Year"] = cleaned_df["Feature_name"].astype(str).str.strip()

x_cols = [f"x{i}" for i in range(1, 67)]
for col in x_cols:
    cleaned_df[col] = pd.to_numeric(cleaned_df[col].astype(str).str.replace('"', '').str.strip(), errors="coerce")

print("\n--- DUPLICATE ANALYSIS ---")
exact_duplicates = cleaned_df.duplicated(subset=x_cols + ["Standard_Label"]).sum()
print(f"Exact duplicated rows (identical features + label): {exact_duplicates}")

company_year_dups = cleaned_df.duplicated(subset=["Company", "Year"], keep=False)
print(f"Total rows involved in duplicate (Company, Year): {company_year_dups.sum()}")

cleaned_df = cleaned_df.drop_duplicates(subset=["Company", "Year", "Target"] + x_cols).reset_index(drop=True)
print(f"Rows after dropping exact duplicates: {len(cleaned_df)}")

cleaned_path = os.path.join(DATA_DIR, "cleaned_dataset.csv")
cleaned_df.to_csv(cleaned_path, index=False)
print(f"Saved cleaned_dataset.csv with {len(cleaned_df)} rows.")

print("\n--- DATA QUALITY REPORT ---")
missing_stats = []
for col in x_cols:
    cnt = cleaned_df[col].isna().sum()
    pct = (cnt / len(cleaned_df)) * 100
    missing_stats.append({
        "Feature": col,
        "Missing_Count": int(cnt),
        "Missing_Pct": round(pct, 2)
    })
missing_df = pd.DataFrame(missing_stats)
missing_df.to_csv(os.path.join(RESULTS_DIR, "missing_values_report.csv"), index=False)
print("Saved results/missing_values_report.csv")

inf_stats = {}
for col in x_cols:
    num_inf = np.isinf(cleaned_df[col]).sum()
    if num_inf > 0:
        inf_stats[col] = int(num_inf)
print(f"Features with Inf values: {inf_stats if inf_stats else 'None'}")

print("\n--- COMPANY-LEVEL LEAKAGE-FREE TRAIN/TEST SPLIT ---")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
groups = cleaned_df["Company"]
y = cleaned_df["Target"]

train_idx, test_idx = next(sgkf.split(cleaned_df, y, groups))
train_df = cleaned_df.iloc[train_idx].copy().reset_index(drop=True)
test_df = cleaned_df.iloc[test_idx].copy().reset_index(drop=True)

train_companies = set(train_df["Company"])
test_companies = set(test_df["Company"])
overlap = train_companies.intersection(test_companies)

print(f"Train rows: {len(train_df)} ({len(train_df)/len(cleaned_df)*100:.1f}%), Unique companies: {len(train_companies)}")
print(f"Test rows: {len(test_df)} ({len(test_df)/len(cleaned_df)*100:.1f}%), Unique companies: {len(test_companies)}")
print(f"Overlapping companies between Train and Test: {len(overlap)}")
assert len(overlap) == 0, "Data leakage detected: overlapping companies!"

print("\nTrain Class Distribution:")
for target_int, class_name in [(0, "Bankrupt"), (1, "Financial Distress"), (2, "Probably Bankrupt"), (3, "Healthy")]:
    cnt = (train_df["Target"] == target_int).sum()
    pct = cnt / len(train_df) * 100
    print(f"  {class_name} ({target_int}): {cnt} ({pct:.2f}%)")

print("\nTest Class Distribution:")
for target_int, class_name in [(0, "Bankrupt"), (1, "Financial Distress"), (2, "Probably Bankrupt"), (3, "Healthy")]:
    cnt = (test_df["Target"] == target_int).sum()
    pct = cnt / len(test_df) * 100
    print(f"  {class_name} ({target_int}): {cnt} ({pct:.2f}%)")

train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
print("Saved data/train.csv and data/test.csv")

label_mapping_info = {
    "string_to_numeric": label_to_numeric,
    "numeric_to_string": {v: k for k, v in label_to_numeric.items()},
    "feature_columns": x_cols,
    "id_columns": ["Company", "Year", "Folder_name", "Feature_name"],
    "target_column": "Target"
}
with open(os.path.join(MODELS_DIR, "label_mapping.json"), "w") as f:
    json.dump(label_mapping_info, f, indent=4)
print("Saved models/label_mapping.json")
