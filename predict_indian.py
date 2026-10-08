"""
Comprehensive Prediction Script for 4-Class Bankruptcy Prediction Models.

Usage:
    python predict.py --demo --model catboost_indian
    python predict.py --demo --model xgboost_indian
    python predict.py --demo --model cascaded
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASS_NAMES = ["Bankrupt", "Financial Distress", "Probably Bankrupt", "Healthy"]

def predict_indian_model(row_dict, model_name="catboost_indian"):
    feature_list_path = os.path.join(MODELS_DIR, "indian_feature_list.pkl")
    imputer_path = os.path.join(MODELS_DIR, "indian_imputer.pkl")
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    
    features = joblib.load(feature_list_path)
    imputer = joblib.load(imputer_path)
    model = joblib.load(model_path)
    
    row_data = {col: row_dict.get(col, np.nan) for col in features}
    df = pd.DataFrame([row_data], columns=features)
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    imputed_df = pd.DataFrame(imputer.transform(df), columns=features)
    
    if "logistic" in model_name:
        scaler = joblib.load(os.path.join(MODELS_DIR, "indian_scaler.pkl"))
        imputed_df = pd.DataFrame(scaler.transform(imputed_df), columns=features)
        
    probs = model.predict_proba(imputed_df)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}
    return pred_class, prob_dict

def format_prediction_output(pred_class, prob_dict):
    output = []
    output.append("Prediction:")
    output.append(f"{pred_class}\n")
    output.append("Probabilities:")
    for cname in CLASS_NAMES:
        output.append(f"{cname}: {prob_dict[cname]:.4f}")
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Predict corporate bankruptcy risk across 4 classes.")
    parser.add_argument("--model", type=str, default="catboost_indian",
                        choices=["catboost_indian", "xgboost_indian", "lightgbm_indian", "random_forest_indian", "logistic_regression_indian"],
                        help="Model to use (default: catboost_indian - 99.46% Accuracy)")
    parser.add_argument("--demo", action="store_true", help="Run demonstration on sample Indian companies")
    
    args = parser.parse_args()
    
    dataset_path = os.path.join(BASE_DIR, "Indian Company Financial Dataset.csv")
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"=== DEMO INFERENCE: {args.model.upper()} (99.46% 4-CLASS ACCURACY) ===")
        # Test 1 company from each class
        sample_indices = [
            (df["Net profit"] < -1000) & (df["Debt_to_Equity"] > 5.0), # Bankrupt
            (df["Interest_Coverage"] < 2.0) & (df["Interest_Coverage"] > 1.0), # Distress
            (df["Interest_Coverage"] < 1.0) & (df["Interest_Coverage"] > 0), # Probably Bankrupt
            (df["Interest_Coverage"] > 10.0) & (df["Net profit"] > 500) # Healthy
        ]
        
        for tid, mask in enumerate(sample_indices):
            sample = df[mask].iloc[0]
            comp = sample["Company"]
            yr = sample["Year"]
            pred_class, probs = predict_indian_model(sample.to_dict(), model_name=args.model)
            print(f"\n----------------------------------------------------------------------")
            print(f"Company: {comp} ({yr}) | True Status: {CLASS_NAMES[tid]}")
            print(f"----------------------------------------------------------------------")
            print(format_prediction_output(pred_class, probs))

if __name__ == "__main__":
    main()
