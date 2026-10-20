import os
import joblib
import pandas as pd
import numpy as np
import shap
from flask import Flask, request, jsonify, render_template

from live_data_fetcher import fetch_and_predict, normalize_ticker

# ---------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------
app = Flask(__name__, template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "Indian Company Financial Dataset.csv")

CLASS_NAMES = ["Bankrupt", "Financial Distress", "Probably Bankrupt", "Healthy"]

# Essential, core financial features only (eliminates minor line-item noise)
ESSENTIAL_FEATURES = [
    "Interest_Coverage",
    "Debt_to_Equity",
    "Operating_Margin",
    "Net profit",
    "Operating Profit",
    "Borrowings",
    "ROA",
    "ROE",
    "Cash_to_Debt",
    "Working_Capital_Ratio",
    "Cash from Operating Activity",
    "Borrowing_Growth",
    "Revenue_Growth",
    "Profit_Growth"
]

FEATURE_DISPLAY_NAMES = {
    "Interest_Coverage": "Interest Coverage Ratio",
    "Debt_to_Equity": "Debt-to-Equity Ratio",
    "Operating_Margin": "Operating Margin",
    "ROA": "Return on Assets (ROA)",
    "ROE": "Return on Equity (ROE)",
    "Working_Capital_Ratio": "Working Capital Ratio",
    "Cash_to_Debt": "Cash-to-Debt Ratio",
    "Net profit": "Net Profit",
    "Operating Profit": "Operating Profit (EBIT)",
    "Borrowings": "Total Debt / Borrowings",
    "Cash from Operating Activity": "Operating Cash Flow",
    "Borrowing_Growth": "Debt Growth (YoY)",
    "Revenue_Growth": "Sales Growth (YoY)",
    "Profit_Growth": "Net Profit Growth (YoY)",
    "Sales": "Sales Revenue"
}

print("[+] Loading trained models and reference datasets...")
features = joblib.load(os.path.join(MODELS_DIR, "indian_feature_list.pkl"))
imputer = joblib.load(os.path.join(MODELS_DIR, "indian_imputer.pkl"))
model = joblib.load(os.path.join(MODELS_DIR, "catboost_indian.pkl"))
dataset_df = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else None
explainer = shap.TreeExplainer(model)
print("[+] Assets loaded successfully.")

# ---------------------------------------------------------
# Essential Dynamic Formatting & Explanation
# ---------------------------------------------------------
def format_val(name: str, val: float) -> str:
    if val is None or np.isnan(val):
        return "N/A"
    name_l = name.lower()
    if any(k in name_l for k in ["coverage", "debt_to_equity", "cash_to_debt"]):
        return f"{val:.2f}x"
    if any(k in name_l for k in ["margin", "roa", "roe", "growth"]) or ("ratio" in name_l and "coverage" not in name_l):
        return f"{val:.1%}" if abs(val) < 10 else f"{val:.2f}"
    if any(k in name_l for k in ["sales", "profit", "borrowings", "cash", "reserves"]):
        return f"₹{val:,.1f} Cr"
    return f"{val:.2f}"

def get_essential_explanation(name: str, val: float, is_risk: bool) -> str:
    """Concise, high-impact plain-English explanation for essential financial factors."""
    if name == "Interest_Coverage":
        if val < 0:
            return "Operating loss makes it impossible to pay interest on borrowings (acute default risk)."
        elif val < 1.0:
            return f"Operating profit cannot pay the annual interest on its debt ({val:.2f}x is below 1.0x)."
        elif val < 2.0:
            return f"Tight debt servicing coverage of {val:.2f}x leaves thin safety margins."
        else:
            return f"Operating profit comfortably pays interest on debt by {val:.2f}x."

    elif name == "Debt_to_Equity":
        if val >= 10.0 or val < 0:
            return "Extreme leverage or negative equity: borrowings are many times the owners' funds."
        elif val > 3.0:
            return f"High leverage ({val:.2f}x): heavy borrowings create high refinancing risk."
        elif val > 1.5:
            return f"Moderate leverage of {val:.2f}x within standard industrial debt limits."
        else:
            return f"Low leverage of {val:.2f}x provides a strong equity cushion against financial stress."

    elif name == "Operating_Margin":
        if val < 0:
            return f"The core business is making an operating loss ({val:.1%})."
        elif val < 0.05:
            return f"Thin operating margin of {val:.1%} provides little buffer against cost increases."
        else:
            return f"Healthy operating margin of {val:.1%} shows solid pricing power and cost efficiency."

    elif name == "Net profit":
        if val < 0:
            return f"Substantial net loss of {format_val(name, val)} actively depleting equity."
        else:
            return f"Solid net profit of {format_val(name, val)} strengthening capital reserves."

    elif name == "Operating Profit":
        if val < 0:
            return f"Operating loss of {format_val(name, val)} from day-to-day business operations."
        else:
            return f"Strong operating profit of {format_val(name, val)} generating debt servicing cash flow."

    elif name == "ROA":
        if val < 0:
            return f"Negative return on assets ({val:.1%}) eroding capital investments."
        else:
            return f"Solid return on assets of {val:.1%} generating productive capital yield."

    elif name == "ROE":
        if val < 0:
            return f"Negative return on equity ({val:.1%}) wiping out shareholder book value."
        else:
            if is_risk:
                return f"Subdued or distorted return on equity ({val:.1%}) under heavy balance-sheet leverage."
            return f"Strong return on equity of {val:.1%} compounding shareholder capital."

    elif name == "Borrowings":
        if is_risk:
            return f"Heavy total borrowings of {format_val(name, val)} creating high recurring finance costs."
        else:
            return f"Manageable borrowings of {format_val(name, val)} well aligned with cash generation."

    elif name == "Cash_to_Debt":
        if val < 0.1:
            return f"Cash covers less than 10% of debt ({val:.2f}x), signaling thin liquidity."
        else:
            return f"Cash cushion covers {val:.2f}x of total debt, buffering debt maturities."

    elif name == "Working_Capital_Ratio":
        if val < 0:
            return f"Negative working capital ({val:.2f}): current liabilities exceed liquid assets."
        else:
            return f"Healthy working capital ratio ({val:.2f}) funding daily operations."

    elif name == "Cash from Operating Activity":
        if val < 0:
            return f"Negative operating cash flow ({format_val(name, val)}): core business is burning cash."
        else:
            return f"Positive operating cash flow of {format_val(name, val)} self-funding debt service."

    elif name == "Revenue_Growth":
        if val < 0:
            return f"Sales contracting by {val:.1%}, reducing cash available for debt repayment."
        else:
            return f"Sales expanding by +{val:.1%}, broadening cash flow capacity."

    elif name == "Borrowing_Growth":
        if val > 0.15:
            return f"Rapid borrowing growth (+{val:.1%}) accelerating debt load."
        else:
            return f"Disciplined debt growth ({val:.1%}) keeping leverage controlled."

    return f"Level of {FEATURE_DISPLAY_NAMES.get(name, name)} ({format_val(name, val)}) influencing solvency."

def extract_essential_factors(all_shap_df: pd.DataFrame, pred_class: str):
    """Filters to ESSENTIAL features with meaningful SHAP impact (|SHAP| >= 0.04)."""
    is_healthy = (pred_class == "Healthy")
    
    # Restrict to essential financial factors only
    df_ess = all_shap_df[all_shap_df["Feature"].isin(ESSENTIAL_FEATURES)].copy()
    
    risk_list = []
    health_list = []

    if is_healthy:
        # For Healthy: Significant positive SHAP represents key health pillars
        h_cand = df_ess[df_ess["SHAP_Value"] >= 0.04].sort_values("SHAP_Value", ascending=False).head(4)
        for _, r in h_cand.iterrows():
            feat, val, s_val = r["Feature"], float(r["Actual_Value"]), float(r["SHAP_Value"])
            health_list.append({
                "feature": feat,
                "name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "actual": format_val(feat, val),
                "impact": f"+{s_val:.2f} toward Healthy",
                "shap_raw": round(s_val, 4),
                "explanation": get_essential_explanation(feat, val, is_risk=False)
            })

        # Only list risk if there is an actual significant risk feature (|SHAP| >= 0.05)
        r_cand = df_ess[df_ess["SHAP_Value"] <= -0.05].sort_values("SHAP_Value", ascending=True).head(3)
        for _, r in r_cand.iterrows():
            feat, val, s_val = r["Feature"], float(r["Actual_Value"]), float(r["SHAP_Value"])
            risk_list.append({
                "feature": feat,
                "name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "actual": format_val(feat, val),
                "impact": f"{s_val:.2f}",
                "shap_raw": round(s_val, 4),
                "explanation": get_essential_explanation(feat, val, is_risk=True)
            })
    else:
        # For Distressed / Bankrupt: Significant positive SHAP represents major failure drivers
        r_cand = df_ess[df_ess["SHAP_Value"] >= 0.04].sort_values("SHAP_Value", ascending=False).head(4)
        for _, r in r_cand.iterrows():
            feat, val, s_val = r["Feature"], float(r["Actual_Value"]), float(r["SHAP_Value"])
            risk_list.append({
                "feature": feat,
                "name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "actual": format_val(feat, val),
                "impact": f"+{s_val:.2f}",
                "shap_raw": round(s_val, 4),
                "explanation": get_essential_explanation(feat, val, is_risk=True)
            })

        # Only list health if there is an actual meaningful mitigating buffer
        h_cand = df_ess[df_ess["SHAP_Value"] <= -0.05].sort_values("SHAP_Value", ascending=True).head(2)
        for _, r in h_cand.iterrows():
            feat, val, s_val = r["Feature"], float(r["Actual_Value"]), float(r["SHAP_Value"])
            health_list.append({
                "feature": feat,
                "name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "actual": format_val(feat, val),
                "impact": f"+{abs(s_val):.2f} buffer",
                "shap_raw": round(s_val, 4),
                "explanation": get_essential_explanation(feat, val, is_risk=False)
            })

    return risk_list, health_list

def generate_dynamic_summary(pred_class: str, conf: float, ratios: dict, top_risk: list, top_health: list) -> str:
    ic = ratios.get("Interest Coverage", "N/A")
    de = ratios.get("Debt-to-Equity", "N/A")
    om = ratios.get("Operating Margin", "N/A")

    if pred_class == "Healthy":
        return (
            f"The model classified this entity as <b>Healthy</b> with <b>{conf:.2f}% confidence</b>. "
            f"Operating profits comfortably cover interest obligations ({ic}x) alongside low financial leverage ({de}x) "
            f"and positive operating profitability ({om}), ensuring strong solvency resilience."
        )
    elif pred_class == "Financial Distress":
        return (
            f"The model classified this entity under <b>Financial Distress</b> with <b>{conf:.2f}% confidence</b>. "
            f"Debt servicing capacity is compressed ({ic}x) with elevated leverage ({de}x) and tight margins ({om}), "
            f"indicating narrowing liquidity buffers that require close credit monitoring."
        )
    elif pred_class == "Probably Bankrupt":
        return (
            f"The model classified this entity as <b>Probably Bankrupt</b> with <b>{conf:.2f}% confidence</b>. "
            f"Operating profits cannot cover interest expenses ({ic}x) while balance-sheet leverage is extreme ({de}x) "
            f"and core operations are incurring losses ({om}), creating severe default probability."
        )
    else:
        return (
            f"The model classified this entity as <b>Bankrupt</b> with <b>{conf:.2f}% confidence</b>. "
            f"Severe balance sheet insolvency with negative or depleted equity ({de}x), negative interest coverage ({ic}x), "
            f"and persistent operational losses ({om}) signify acute default and distressed liquidation risk."
        )

def get_essential_snapshot(ratios: dict):
    """Returns the 6 essential solvency indicators with clear benchmark status."""
    def _parse(val_str, default=0.0):
        try:
            clean = str(val_str).replace("₹", "").replace("Cr", "").replace("x", "").replace("%", "").replace(",", "").strip()
            return float(clean)
        except Exception:
            return default

    ic = _parse(ratios.get("Interest Coverage", 0.0))
    de = _parse(ratios.get("Debt-to-Equity", 0.0))
    om = _parse(ratios.get("Operating Margin", 0.0)) / (100.0 if "%" in str(ratios.get("Operating Margin", "")) else 1.0)
    roa = _parse(ratios.get("Return on Assets (ROA)", 0.0)) / (100.0 if "%" in str(ratios.get("Return on Assets (ROA)", "")) else 1.0)
    roe = _parse(ratios.get("Return on Equity (ROE)", 0.0)) / (100.0 if "%" in str(ratios.get("Return on Equity (ROE)", "")) else 1.0)
    wc = _parse(ratios.get("Working Capital Ratio", 0.0))

    return [
        {"metric": "Interest Coverage Ratio", "value": f"{ic:.2f}x", "benchmark": "≥ 2.50x", "status": "Strong" if ic >= 2.5 else "Moderate" if ic >= 1.5 else "Concerning" if ic >= 1.0 else "Critical", "color": "green" if ic >= 2.5 else "yellow" if ic >= 1.5 else "orange" if ic >= 1.0 else "red", "implication": "Operating profit covers interest obligations" if ic >= 2.5 else "Operating profit cannot cover interest expense"},
        {"metric": "Debt-to-Equity Ratio", "value": f"{de:.2f}x", "benchmark": "≤ 1.50x", "status": "Critical" if (de < 0 or de >= 10.0) else "Concerning" if de > 3.0 else "Moderate" if de > 1.5 else "Strong", "color": "red" if (de < 0 or de >= 10.0) else "orange" if de > 3.0 else "yellow" if de > 1.5 else "green", "implication": "Conservative debt with solid equity buffer" if de <= 1.5 and de > 0 else "Excessive leverage / negative equity"},
        {"metric": "Operating Margin", "value": f"{om:.1%}", "benchmark": "≥ 12.0%", "status": "Strong" if om >= 0.12 else "Moderate" if om >= 0.05 else "Concerning" if om >= 0.0 else "Critical", "color": "green" if om >= 0.12 else "yellow" if om >= 0.05 else "orange" if om >= 0.0 else "red", "implication": "Core business profitable and self-sustaining" if om >= 0.05 else "Core operations incurring operating loss"},
        {"metric": "Return on Assets (ROA)", "value": f"{roa:.1%}", "benchmark": "≥ 5.0%", "status": "Strong" if roa >= 0.05 else "Moderate" if roa >= 0.01 else "Concerning" if roa >= 0.0 else "Critical", "color": "green" if roa >= 0.05 else "yellow" if roa >= 0.01 else "orange" if roa >= 0.0 else "red", "implication": "Productive asset utilization" if roa >= 0.01 else "Asset base generating operational loss"},
        {"metric": "Return on Equity (ROE)", "value": f"{roe:.1%}", "benchmark": "≥ 12.0%", "status": "Strong" if roe >= 0.12 else "Moderate" if roe >= 0.05 else "Concerning" if roe >= 0.0 else "Critical", "color": "green" if roe >= 0.12 else "yellow" if roe >= 0.05 else "orange" if roe >= 0.0 else "red", "implication": "Compounding shareholder equity" if roe >= 0.05 else "Shareholder capital erosion"},
        {"metric": "Working Capital Ratio", "value": f"{wc:.2f}", "benchmark": "≥ 0.15", "status": "Strong" if wc >= 0.15 else "Moderate" if wc >= 0.05 else "Concerning" if wc >= 0.0 else "Critical", "color": "green" if wc >= 0.15 else "yellow" if wc >= 0.05 else "orange" if wc >= 0.0 else "red", "implication": "Liquid short-term capital" if wc >= 0.05 else "Current liabilities exceed liquid assets"},
    ]

# ---------------------------------------------------------
# Flask Endpoints
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/companies", methods=["GET"])
def get_companies():
    if dataset_df is not None:
        comps = sorted(dataset_df["Company"].unique().tolist())
        return jsonify({"companies": comps})
    return jsonify({"companies": []})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    mode = data.get("mode", "live")
    
    try:
        if mode == "live":
            ticker = data.get("ticker", "SUZLON.NS")
            res = fetch_and_predict(ticker, active_model=model, explainer=explainer)
            all_shap_df = res.get("all_shap_df", res["shap_df"])
            ratios = res["ratios"]
            pred = res["prediction"]
            conf = res["confidence"]
            probs = res["probabilities"]
            name = f"{res['company_name']} ({res['ticker']})"
            subtitle = f"Sector: {res['sector']} | Market Price: ₹{res['current_price']:,.2f} | Latest Filing: {res['period']}"
            
        elif mode == "historical":
            comp_name = data.get("company", "ALOK INDUSTRIES LTD")
            comp_df = dataset_df[dataset_df["Company"] == comp_name].sort_values(by="Year").reset_index(drop=True)
            if comp_df.empty:
                return jsonify({"error": f"Company '{comp_name}' not found."}), 404
            latest_row = comp_df.iloc[-1]
            
            row_data = pd.DataFrame([{col: latest_row.get(col, np.nan) for col in features}])
            for col in features:
                row_data[col] = pd.to_numeric(row_data[col], errors="coerce")
            imputed_data = pd.DataFrame(imputer.transform(row_data), columns=features)
            probs_arr = model.predict_proba(imputed_data)[0]
            pred_idx = int(np.argmax(probs_arr))
            pred = CLASS_NAMES[pred_idx]
            conf = round(float(probs_arr[pred_idx]) * 100, 2)
            probs = {CLASS_NAMES[i]: round(float(probs_arr[i]) * 100, 2) for i in range(4)}
            
            sv = explainer(imputed_data)
            vals = sv.values[0, :, pred_idx] if len(sv.values.shape) == 3 else sv.values[0, :]
            all_shap_df = pd.DataFrame({
                "Feature": features,
                "SHAP_Value": [float(vals[i]) for i in range(len(features))],
                "Actual_Value": [float(imputed_data.iloc[0, i]) for i in range(len(features))]
            })
            
            ic = float(latest_row.get("Interest_Coverage", 0.0))
            de = float(latest_row.get("Debt_to_Equity", 0.0))
            om = float(latest_row.get("Operating_Margin", 0.0))
            roa = float(latest_row.get("ROA", 0.0))
            roe = float(latest_row.get("ROE", 0.0))
            wc = float(latest_row.get("Working_Capital_Ratio", 0.0))
            
            name = comp_name
            subtitle = f"Audited Historical Filing: Fiscal Year {int(latest_row['Year'])}"
            ratios = {
                "Sales (₹ Cr)": f"{latest_row.get('Sales', 0.0):.1f}",
                "Net Profit (₹ Cr)": f"{latest_row.get('Net profit', 0.0):.1f}",
                "Total Debt (₹ Cr)": f"{latest_row.get('Borrowings', 0.0):.1f}",
                "Interest Coverage": f"{ic:.2f}",
                "Debt-to-Equity": f"{de:.2f}",
                "Operating Margin": f"{om:.1%}",
                "Return on Assets (ROA)": f"{roa:.1%}",
                "Return on Equity (ROE)": f"{roe:.1%}",
                "Working Capital Ratio": f"{wc:.2f}"
            }
            
        else: # custom
            s_sales = float(data.get("sales", 5000.0))
            s_profit = float(data.get("profit", 600.0))
            s_debt = float(data.get("debt", 400.0))
            s_eq = float(data.get("equity", 2600.0))
            s_ic = float(data.get("ic", 20.0))
            s_de = float(data.get("de", 0.15))
            s_om = float(data.get("om", 0.20))
            
            custom_input = {f: np.nan for f in features}
            custom_input.update({
                "Sales": s_sales, "Net profit": s_profit, "Borrowings": s_debt,
                "Equity Share Capital": s_eq * 0.25, "Reserves": s_eq * 0.75,
                "Debt_to_Equity": s_de, "Interest_Coverage": s_ic, "Operating_Margin": s_om
            })
            c_row = pd.DataFrame([custom_input])
            c_imp = pd.DataFrame(imputer.transform(c_row), columns=features)
            c_probs = model.predict_proba(c_imp)[0]
            c_idx = int(np.argmax(c_probs))
            pred = CLASS_NAMES[c_idx]
            conf = round(float(c_probs[c_idx]) * 100, 2)
            probs = {CLASS_NAMES[i]: round(float(c_probs[i]) * 100, 2) for i in range(4)}
            
            c_sv = explainer(c_imp)
            c_vals = c_sv.values[0, :, c_idx] if len(c_sv.values.shape) == 3 else c_sv.values[0, :]
            all_shap_df = pd.DataFrame({
                "Feature": features,
                "SHAP_Value": [float(c_vals[i]) for i in range(len(features))],
                "Actual_Value": [float(c_imp.iloc[0, i]) for i in range(len(features))]
            })
            name = "Custom Simulation"
            subtitle = "Hypothetical Balance Sheet & Solvency Numbers"
            ratios = {
                "Sales (₹ Cr)": f"{s_sales:.1f}",
                "Net Profit (₹ Cr)": f"{s_profit:.1f}",
                "Total Debt (₹ Cr)": f"{s_debt:.1f}",
                "Interest Coverage": f"{s_ic:.2f}",
                "Debt-to-Equity": f"{s_de:.2f}",
                "Operating Margin": f"{s_om:.1%}",
                "Return on Assets (ROA)": f"{float(c_imp.iloc[0]['ROA']):.1%}",
                "Return on Equity (ROE)": f"{float(c_imp.iloc[0]['ROE']):.1%}",
                "Working Capital Ratio": f"{float(c_imp.iloc[0]['Working_Capital_Ratio']):.2f}"
            }

        # Extract only essential high-impact factors
        risk_factors, health_factors = extract_essential_factors(all_shap_df, pred)
        summary = generate_dynamic_summary(pred, conf, ratios, risk_factors, health_factors)
        snapshot = get_essential_snapshot(ratios)

        # Build clean SHAP chart for essential factors only
        is_healthy = (pred == "Healthy")
        shap_chart_items = []
        
        # Display the key decision factors in the chart
        chart_features = health_factors + risk_factors
        for item in chart_features:
            s_val = item["shap_raw"]
            is_good = (s_val > 0 and is_healthy) or (s_val < 0 and not is_healthy)
            shap_chart_items.append({
                "feature": item["name"],
                "shap_value": s_val,
                "actual_value": item["actual"],
                "color": "#10B981" if is_good else "#EF4444"
            })

        return jsonify({
            "name": name,
            "subtitle": subtitle,
            "prediction": pred,
            "confidence": conf,
            "probabilities": probs,
            "ratios": ratios,
            "summary": summary,
            "risk_factors": risk_factors,
            "health_factors": health_factors,
            "snapshot": snapshot,
            "shap_chart": shap_chart_items
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("[+] Launching Flask Web Application on http://localhost:5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
