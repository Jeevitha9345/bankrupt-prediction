"""
Standalone Prediction Script for 4-Class Bankruptcy Prediction Model.

Usage examples:
    python predict.py --demo --model cascaded
    python predict.py --demo --model enhanced_xgboost
    python predict.py --csv sample_input.csv --model cascaded
    python predict.py --json "{\"x1\": 0.45, \"x2\": 0.22, \"x3\": 0.05, ...}"
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

MODEL_CHOICES = {
    "cascaded": "cascaded",
    "enhanced_xgboost": "enhanced_xgboost.pkl",
    "xgboost": "xgboost.pkl",
    "random_forest": "random_forest.pkl",
    "catboost": "catboost.pkl",
    "lightgbm": "lightgbm.pkl",
    "logistic_regression": "logistic_regression.pkl"
}

CLASS_NAMES = ["Bankrupt", "Financial Distress", "Probably Bankrupt", "Healthy"]
RAW_FEATURES = [f"x{i}" for i in range(1, 67)]

def engineer_features(df):
    d = df.copy()
    d["num_negative"] = (d[RAW_FEATURES] < 0).sum(axis=1)
    d["num_zero"] = (d[RAW_FEATURES] == 0).sum(axis=1)
    d["mean_ratio"] = d[RAW_FEATURES].mean(axis=1)
    d["std_ratio"] = d[RAW_FEATURES].std(axis=1)
    d["median_ratio"] = d[RAW_FEATURES].median(axis=1)
    d["x1_div_x2"] = d["x1"] / (d["x2"].abs() + 1e-4)
    d["x3_mul_x5"] = d["x3"] * d["x5"]
    d["x24_div_x45"] = d["x24"] / (d["x45"].abs() + 1e-4)
    d["x13_div_x5"] = d["x13"] / (d["x5"].abs() + 1e-4)
    return d

def predict_single(feature_dict, model_name="cascaded"):
    imputer = joblib.load(os.path.join(MODELS_DIR, "imputer.pkl"))
    
    # Construct 1-row DataFrame aligned with training features
    row_data = {col: feature_dict.get(col, np.nan) for col in RAW_FEATURES}
    df = pd.DataFrame([row_data], columns=RAW_FEATURES)
    for col in RAW_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    imputed_df = pd.DataFrame(imputer.transform(df), columns=RAW_FEATURES)
    
    if model_name.lower() == "cascaded":
        eng_df = engineer_features(imputed_df)
        gate_model = joblib.load(os.path.join(MODELS_DIR, "solvency_gate_model.pkl"))
        sev_model = joblib.load(os.path.join(MODELS_DIR, "distress_severity_model.pkl"))
        
        p_healthy = gate_model.predict_proba(eng_df)[0][1] # P(Healthy)
        p_distress_overall = 1.0 - p_healthy
        
        p_sev = sev_model.predict_proba(eng_df)[0] # P(0, 1, 2)
        sev_classes = list(sev_model.classes_)
        
        # Combined calibrated 4-class probabilities
        probs = np.zeros(4)
        probs[3] = p_healthy
        for idx, cls in enumerate(sev_classes):
            probs[cls] = p_sev[idx] * p_distress_overall
            
        pred_idx = 3 if p_healthy >= 0.40 else int(sev_classes[np.argmax(p_sev)])
        pred_class = CLASS_NAMES[pred_idx]
        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}
        return pred_class, prob_dict
        
    elif model_name.lower() == "enhanced_xgboost":
        eng_df = engineer_features(imputed_df)
        model = joblib.load(os.path.join(MODELS_DIR, "enhanced_xgboost.pkl"))
        probs = model.predict_proba(eng_df)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}
        return pred_class, prob_dict
        
    else:
        # Standard models
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        model = joblib.load(os.path.join(MODELS_DIR, MODEL_CHOICES[model_name.lower()]))
        
        if model_name.lower() == "logistic_regression":
            proc_df = pd.DataFrame(scaler.transform(imputed_df), columns=RAW_FEATURES)
        else:
            proc_df = imputed_df
            
        probs = model.predict_proba(proc_df)[0]
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
    parser.add_argument("--model", type=str, default="cascaded", choices=list(MODEL_CHOICES.keys()),
                        help="Model to use (default: cascaded - Highest Accuracy Mode)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV containing feature values x1-x66")
    parser.add_argument("--json", type=str, default=None,
                        help="JSON string or path to JSON file containing feature dictionary")
    parser.add_argument("--demo", action="store_true",
                        help="Run demonstration using sample records from test set")
                        
    args = parser.parse_args()
    
    if args.demo or (args.csv is None and args.json is None):
        print(f"=== DEMO PREDICTION USING MODEL: {args.model.upper()} ===")
        test_csv_path = os.path.join(BASE_DIR, "data", "test.csv")
        if os.path.exists(test_csv_path):
            test_df = pd.read_csv(test_csv_path)
            for target_id, target_name in enumerate(CLASS_NAMES):
                sample_rows = test_df[test_df["Target"] == target_id]
                if not sample_rows.empty:
                    sample = sample_rows.iloc[0]
                    comp = sample.get("Company", "Sample Company")
                    yr = sample.get("Year", "N/A")
                    feat_dict = {f"x{i}": sample[f"x{i}"] for i in range(1, 67)}
                    
                    pred_class, probs = predict_single(feat_dict, model_name=args.model)
                    print(f"\n--------------------------------------------------")
                    print(f"Sample Company: {comp} (Year: {yr}) | True Class: {target_name}")
                    print(f"--------------------------------------------------")
                    print(format_prediction_output(pred_class, probs))
        return

    feature_dict = {}
    if args.json:
        if os.path.exists(args.json):
            with open(args.json, "r") as f:
                feature_dict = json.load(f)
        else:
            feature_dict = json.loads(args.json)
    elif args.csv:
        csv_df = pd.read_csv(args.csv)
        feature_dict = csv_df.iloc[0].to_dict()
        
    pred_class, probs = predict_single(feature_dict, model_name=args.model)
    print(format_prediction_output(pred_class, probs))

if __name__ == "__main__":
    main()
