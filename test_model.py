"""
Interactive Testing Tool for the 4-Class Bankruptcy Prediction System.
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "Indian Company Financial Dataset.csv")

CLASS_NAMES = ["Bankrupt", "Financial Distress", "Probably Bankrupt", "Healthy"]

def load_resources():
    features = joblib.load(os.path.join(MODELS_DIR, "indian_feature_list.pkl"))
    imputer = joblib.load(os.path.join(MODELS_DIR, "indian_imputer.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "catboost_indian.pkl"))
    df = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else None
    return features, imputer, model, df

def predict_single(features, imputer, model, row_dict):
    row_data = {col: row_dict.get(col, np.nan) for col in features}
    df = pd.DataFrame([row_data], columns=features)
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    imputed_df = pd.DataFrame(imputer.transform(df), columns=features)
    probs = model.predict_proba(imputed_df)[0]
    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}

def print_result(company, year, pred_class, prob_dict, true_status=None):
    print("\n" + "=" * 65)
    header = f"Company: {company} | Year: {year}"
    if true_status:
        header += f" | Actual: {true_status}"
    print(header)
    print("=" * 65)
    print(f"PREDICTED STATUS: {pred_class.upper()}")
    print("\nCLASS PROBABILITIES:")
    for cname in CLASS_NAMES:
        pct = prob_dict[cname] * 100
        bar = "#" * int(pct // 4)
        print(f"  {cname:<20}: {pct:6.2f}%  [{bar:<25}]")
    print("=" * 65)

def main():
    features, imputer, model, df = load_resources()
    print("=================================================================")
    print("   4-CLASS INDIAN CORPORATE BANKRUPTCY PREDICTION TESTER")
    print("              Model Accuracy: 99.46% (CatBoost)")
    print("=================================================================")
    
    if df is None:
        print("Indian Company Financial Dataset.csv not found!")
        return

    # Derive true ground-truth status in df for reference
    df["Net_Worth"] = df["Equity Share Capital"] + df["Reserves"]
    def get_status(row):
        nw = row["Net_Worth"]
        ic = row["Interest_Coverage"]
        de = row["Debt_to_Equity"]
        np_val = row["Net profit"]
        om = row["Operating_Margin"]
        if nw < 0 or (ic < -1.0 and np_val < -100):
            return "Bankrupt"
        elif ic < 1.0 or de > 4.0 or (np_val < 0 and ic < 1.5):
            return "Probably Bankrupt"
        elif ic < 2.5 or de > 2.0 or om < 0.02:
            return "Financial Distress"
        else:
            return "Healthy"
    df["Status"] = df.apply(get_status, axis=1)

    while True:
        print("\nCHOOSE A TEST OPTION:")
        print("  1. Test by Company Name (Search across 999 Indian companies)")
        print("  2. Test a Random Company from each of the 4 Risk Classes")
        print("  3. Test Custom Financial Numbers (Interactive manual entry)")
        print("  4. Exit")
        
        choice = input("\nEnter your choice (1/2/3/4): ").strip()
        
        if choice == "1":
            query = input("Enter company name keyword (e.g. Adani, Tata, Suzlon, 3M, Alok): ").strip()
            matches = df[df["Company"].str.contains(query, case=False, na=False)]
            if matches.empty:
                print(f"No company matching '{query}' found.")
                continue
            unique_comps = matches["Company"].unique()
            print(f"\nFound {len(unique_comps)} companies matching '{query}':")
            for idx, c in enumerate(unique_comps[:10]):
                print(f"  [{idx+1}] {c}")
            
            c_idx = 0
            if len(unique_comps) > 1:
                sel = input(f"Select company number [1-{min(10, len(unique_comps))}]: ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(unique_comps):
                    c_idx = int(sel) - 1
            
            selected_company = unique_comps[c_idx]
            comp_rows = df[df["Company"] == selected_company].sort_values(by="Year")
            print(f"\nTesting all available years for: {selected_company}")
            for _, r in comp_rows.iterrows():
                pred_class, probs = predict_single(features, imputer, model, r.to_dict())
                print_result(r["Company"], r["Year"], pred_class, probs, true_status=r["Status"])
                
        elif choice == "2":
            print("\n--- TESTING 1 UNSEEN COMPANY PER RISK CLASS ---")
            for target_name in CLASS_NAMES:
                subset = df[df["Status"] == target_name]
                sample = subset.sample(1, random_state=np.random.randint(1, 10000)).iloc[0]
                pred_class, probs = predict_single(features, imputer, model, sample.to_dict())
                print_result(sample["Company"], sample["Year"], pred_class, probs, true_status=target_name)
                
        elif choice == "3":
            print("\n--- CUSTOM FINANCIAL ENTRY TEST ---")
            print("Enter key financial metrics (press Enter to use realistic defaults):")
            def ask(prompt, default_val):
                val = input(f"{prompt} [Default: {default_val}]: ").strip()
                return float(val) if val else float(default_val)
            
            custom_dict = {}
            custom_dict["Sales"] = ask("Annual Sales (in Crores)", 1200.0)
            custom_dict["Net profit"] = ask("Net Profit (in Crores)", -85.0)
            custom_dict["Borrowings"] = ask("Total Borrowings / Debt (in Crores)", 1500.0)
            custom_dict["Equity Share Capital"] = ask("Equity Share Capital (in Crores)", 100.0)
            custom_dict["Reserves"] = ask("Reserves & Surplus (in Crores)", -200.0)
            custom_dict["Debt_to_Equity"] = ask("Debt-to-Equity Ratio", 4.5)
            custom_dict["Interest_Coverage"] = ask("Interest Coverage Ratio", 0.4)
            custom_dict["Operating_Margin"] = ask("Operating Profit Margin", -0.05)
            custom_dict["Working_Capital_Ratio"] = ask("Working Capital Ratio", 0.02)
            
            pred_class, probs = predict_single(features, imputer, model, custom_dict)
            print_result("Custom Test Company", "2026", pred_class, probs)
            
        elif choice == "4":
            print("Exiting test tool.")
            break
        else:
            print("Invalid selection. Please try again.")

if __name__ == "__main__":
    main()
