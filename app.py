import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap

from live_data_fetcher import fetch_and_predict, normalize_ticker

# ---------------------------------------------------------
# 1. Page Configuration & Colorful Global Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Corporate Bankruptcy Risk & Explainable AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Global font & light theme enforcement */
    .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 45%, #2563EB 85%, #4F46E5 100%);
        padding: 26px 30px;
        border-radius: 16px;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.4);
        text-align: center;
    }
    .header-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.28);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        color: #93C5FD;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .header-title {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-desc {
        color: #BFDBFE;
        font-size: 1.02rem;
        margin: 8px 0 0 0;
        max-width: 860px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.5;
    }

    /* Primary Button Styling */
    .stButton>button {
        width: 100%;
        height: 48px;
        font-size: 1.05rem;
        font-weight: 700;
        border-radius: 10px;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.5);
        transform: translateY(-1px);
    }

    /* Verdict Card Styles */
    .verdict-box {
        padding: 22px;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    .verdict-healthy {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 50%, #A7F3D0 100%);
        border: 2px solid #10B981;
    }
    .verdict-distress {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 50%, #FDE68A 100%);
        border: 2px solid #F59E0B;
    }
    .verdict-prob-bankrupt {
        background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 50%, #FED7AA 100%);
        border: 2px solid #F97316;
    }
    .verdict-bankrupt {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 50%, #FECACA 100%);
        border: 2px solid #EF4444;
    }

    /* Dynamic Executive Summary Box */
    .summary-box {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #BFDBFE;
        border-left: 6px solid #2563EB;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 22px;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
    }

    /* Metric Tiles */
    .metric-card {
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        background: #FFFFFF;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }

    /* Risk & Health Factor Containers */
    .factor-container-risk {
        background: linear-gradient(180deg, #FFF1F2 0%, #FFFFFF 100%);
        border: 1.5px solid #FECDD3;
        border-top: 5px solid #E11D48;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(225, 29, 72, 0.06);
    }
    .factor-container-health {
        background: linear-gradient(180deg, #F0FDF4 0%, #FFFFFF 100%);
        border: 1.5px solid #BBF7D0;
        border-top: 5px solid #16A34A;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.06);
    }
    .factor-item-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    /* Snapshot Table */
    .snapshot-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
        font-size: 0.92rem;
    }
    .snapshot-table th {
        background: #1E293B;
        color: #F8FAFC;
        padding: 12px 16px;
        font-weight: 700;
        text-align: left;
        letter-spacing: 0.03em;
    }
    .snapshot-table td {
        padding: 11px 16px;
        border-bottom: 1px solid #F1F5F9;
        color: #334155;
    }
    .snapshot-table tr:last-child td {
        border-bottom: none;
    }
    .snapshot-table tr:nth-child(even) {
        background: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Directory Setup & Asset Loading
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "Indian Company Financial Dataset.csv")

CLASS_NAMES = ["Bankrupt", "Financial Distress", "Probably Bankrupt", "Healthy"]

CLASS_COLORS = {
    "Healthy": "#10B981",
    "Financial Distress": "#F59E0B",
    "Probably Bankrupt": "#F97316",
    "Bankrupt": "#EF4444"
}

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
    "Asset_Growth": "Asset Growth (YoY)",
    "Operating_CF_Growth": "Cash Flow Growth (YoY)",
    "Sales": "Sales Revenue",
    "Total Assets": "Total Assets",
    "Total Liabilities": "Total Liabilities",
    "Cash & Bank": "Cash & Liquid Balances",
    "Equity Share Capital": "Equity Share Capital",
    "Reserves": "Net Reserves & Surplus",
    "Profit_Margin": "Net Profit Margin",
    "Expense_Ratio": "Operating Expense Ratio",
    "Depreciation_Ratio": "Depreciation to Assets",
    "Asset_Utilization": "Asset Turnover / Utilization",
    "Receivable_Ratio": "Receivables to Sales",
    "Inventory_Ratio": "Inventory to Sales",
    "Operating_CF_Ratio": "Operating Cash Flow to Assets",
    "Cash_Margin": "Cash Flow Margin",
    "Net Cash Flow": "Net Cash Flow",
    "Profit before tax": "Profit Before Tax (PBT)",
    "Tax": "Tax Expense / Credit",
    "Depreciation": "Depreciation & Amortization",
    "Interest": "Finance Costs / Interest Expense",
    "Raw Material Cost": "Raw Material Expenses",
    "Employee Cost": "Employee & Personnel Cost",
    "Power and Fuel": "Power & Energy Costs",
    "Selling and admin": "Selling & Admin Expenses",
    "Other Expenses": "Other Operating Expenses",
    "Other Income": "Other Non-Operating Income",
    "Dividend Amount": "Dividends Paid",
    "Net Block": "Fixed Assets / Net Block",
    "Capital Work in Progress": "Capital WIP",
    "Investments": "Strategic Investments",
    "Other Assets": "Other Current / Non-Current Assets",
    "Other Liabilities": "Other Current Liabilities",
    "Receivables": "Trade Receivables",
    "Inventory": "Inventory Holdings",
    "Change in Inventory": "Inventory Change",
    "Revenue_CAGR_3Y": "3-Year Sales CAGR",
    "Profit_CAGR_3Y": "3-Year Net Profit CAGR",
    "Revenue_Volatility": "Sales Volatility",
    "Profit_Volatility": "Profit Volatility",
    "Operating_CF_Volatility": "Cash Flow Volatility",
    "Reserve_Growth": "Reserve Growth (YoY)",
    "Receivable_Growth": "Receivables Growth (YoY)",
    "Inventory_Growth": "Inventory Growth (YoY)",
    "Cash_Growth": "Cash Reserves Growth (YoY)"
}

@st.cache_resource
def load_assets():
    features = joblib.load(os.path.join(MODELS_DIR, "indian_feature_list.pkl"))
    imputer = joblib.load(os.path.join(MODELS_DIR, "indian_imputer.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "catboost_indian.pkl"))
    df = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else None
    explainer = shap.TreeExplainer(model)
    return features, imputer, model, df, explainer

features, imputer, model, df, explainer = load_assets()

# Sidebar (Minimalist)
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=55)
    st.markdown("### 🤖 Model Specification")
    st.markdown("""
    * **Algorithm:** CatBoost Gradient Boosting
    * **Verified Test Accuracy:** **99.46%**
    * **Macro F1 Score:** 0.9934
    * **Training Filings:** 8,991 Audited Filings (999 Firms)
    * **Feature Space:** 63 Solvency & Growth Metrics
    * **Explainability:** Multi-Class TreeSHAP
    """)
    st.divider()
    st.caption("Antigravity AI · Enterprise Credit Risk System")

# ---------------------------------------------------------
# 3. Dynamic Reason Generator & Helper Functions
# ---------------------------------------------------------
def format_feature_value(name: str, val: float) -> str:
    """Safely formats any feature value into human-friendly currency, percentage, or ratio."""
    if val is None or np.isnan(val):
        return "N/A"
    name_l = name.lower()
    if any(k in name_l for k in ["coverage", "debt_to_equity", "cash_to_debt"]):
        return f"{val:.2f}x"
    if any(k in name_l for k in ["margin", "roa", "roe", "growth"]) or ("ratio" in name_l and "coverage" not in name_l):
        if abs(val) < 10:
            return f"{val:.1%}"
        return f"{val:.2f}"
    if any(k in name_l for k in ["sales", "profit", "borrowings", "liabilities", "assets", "cash", "reserves", "cost", "expense", "tax", "income"]):
        return f"₹{val:,.1f} Cr"
    return f"{val:.2f}"

def get_feature_explanation(name: str, val: float, is_risk: bool) -> str:
    """Generates dynamically tailored financial explanations based on actual values and context."""
    friendly = FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " "))
    val_str = format_feature_value(name, val)
    
    if name == "Interest_Coverage":
        if val < 0:
            return "Negative interest coverage indicates core operating loss, meaning current operating profit is completely insufficient to cover debt interest."
        elif val < 1.0:
            return f"Interest coverage of {val:.2f}x is below the 1.0x critical threshold, indicating operating earnings cannot cover annual debt servicing charges."
        elif val < 2.0:
            if is_risk:
                return f"Tight coverage of {val:.2f}x leaves very little safety buffer against potential revenue declines or interest rate increases."
            else:
                return f"Moderate coverage of {val:.2f}x generates adequate operational earnings to service current interest dues."
        else:
            return f"Robust interest coverage of {val:.2f}x demonstrates that operating profits comfortably exceed annual interest burdens by a wide margin."

    elif name == "Debt_to_Equity":
        if val >= 10.0 or val < 0:
            return "Severe over-leverage or negative equity: total borrowings heavily exceed or have wiped out owners' funds, indicating acute insolvency risk."
        elif val > 3.0:
            return f"Very high leverage ({val:.2f}x) signifies borrowings are multiple times total shareholder funds, creating heavy refinancing vulnerability."
        elif val > 1.5:
            if is_risk:
                return f"Above-average leverage of {val:.2f}x heightens exposure to economic downturns and debt servicing strain."
            else:
                return f"Manageable leverage of {val:.2f}x stays within normal industrial capital structure tolerances."
        else:
            return f"Low financial leverage of {val:.2f}x provides a solid shareholder equity cushion against financial stress."

    elif name == "Operating_Margin":
        if val < 0:
            return f"Negative operating margin ({val:.1%}) reveals that the core business activities are currently incurring operating losses."
        elif val < 0.05:
            return f"Thin operating margin of {val:.1%} provides minimal absorption against operational cost escalations."
        else:
            return f"Healthy operating margin of {val:.1%} reflects strong pricing power and solid operational cost efficiency."

    elif name == "ROA":
        if val < 0:
            return f"Negative Return on Assets ({val:.1%}) indicates that corporate capital investments and total assets are generating operational losses."
        elif val < 0.03:
            return f"Subdued asset return ({val:.1%}) reflects modest productive capital efficiency from the total asset base."
        else:
            return f"Solid Return on Assets of {val:.1%} demonstrates efficient asset compounding and productive capital deployment."

    elif name == "ROE":
        if val < 0:
            return f"Negative Return on Equity ({val:.1%}) reveals persistent erosion of shareholder book value and net worth."
        else:
            return f"Positive Return on Equity of {val:.1%} highlights healthy capital returns generated for equity holders."

    elif name == "Cash_to_Debt":
        if val < 0.1:
            return f"Thin cash-to-debt ratio of {val:.2f}x indicates available liquid cash reserves cover under 10% of total borrowings."
        else:
            return f"Dependable cash-to-debt ratio of {val:.2f}x provides a liquid cash cushion to absorb debt maturities."

    elif name == "Working_Capital_Ratio":
        if val < 0:
            return f"Negative working capital ({val:.2f}) indicates current liabilities exceed liquid short-term assets, posing cash squeeze risk."
        else:
            return f"Positive working capital ratio of {val:.2f} supports smooth day-to-day liquidity and operational execution."

    elif name == "Net profit":
        if val < 0:
            return f"Net loss of {val_str} directly depletes balance sheet reserves and erodes corporate net worth."
        else:
            return f"Net profit of {val_str} delivers solid retained earnings that bolster equity and fund debt repayments."

    elif name == "Operating Profit":
        if val < 0:
            return f"Operating loss of {val_str} confirms regular business operations cannot self-fund operating costs."
        else:
            return f"Strong operating profit of {val_str} provides primary operational cash generation to cover debt obligations."

    elif name == "Borrowings":
        if is_risk:
            return f"High total borrowings of {val_str} create elevated ongoing interest servicing burdens and refinancing requirements."
        else:
            return f"Disciplined debt level of {val_str} remains in proportion with the firm's operational cash generation capacity."

    elif name == "Cash from Operating Activity":
        if val < 0:
            return f"Negative operating cash flow ({val_str}) signals operational cash burn, forcing reliance on external borrowing."
        else:
            return f"Positive operating cash flow of {val_str} supplies organic internal liquidity to service liabilities."

    elif name == "Borrowing_Growth":
        if val > 0.15:
            return f"Rapid borrowing expansion (+{val:.1%}) indicates accelerating leverage accumulation outpacing internal cash growth."
        elif val <= 0:
            return f"Controlled debt trajectory ({val:.1%}) reflects ongoing deleveraging and disciplined balance sheet management."
        else:
            return f"Moderate debt growth of {val:.1%} remains in step with operational requirements."

    elif name == "Revenue_Growth":
        if val < 0:
            return f"Contracting revenue ({val:.1%}) diminishes top-line cash generation available for meeting debt service."
        else:
            return f"Revenue expansion (+{val:.1%}) strengthens business scale and broadens debt service capacity."

    elif name == "Profit_Growth":
        if val < 0:
            return f"Declining net profit ({val:.1%}) compresses earnings quality and weakens debt repayment buffers."
        else:
            return f"Net profit growth (+{val:.1%}) improves profitability and capital compounding resilience."

    elif name == "Asset_Growth":
        if val < 0:
            return f"Contracting asset base ({val:.1%}) reflects potential asset impairments or capital divestments."
        else:
            return f"Asset expansion (+{val:.1%}) builds operational capacity and supports business growth."

    elif name == "Operating_CF_Growth":
        if val < 0:
            return f"Declining operating cash flow growth ({val:.1%}) tightens operational cash liquidity."
        else:
            return f"Expanding cash flow generation (+{val:.1%}) reinforces liquidity buffers and improves credit health."

    else:
        if is_risk:
            return f"Reported level of {friendly} ({val_str}) exerts downward pressure on solvency, contributing to financial distress."
        else:
            return f"Favorable positioning in {friendly} ({val_str}) provides balance sheet stability and supports creditworthiness."

ESSENTIAL_FEATURES = [
    "Interest_Coverage", "Debt_to_Equity", "Operating_Margin",
    "Net profit", "Operating Profit", "Borrowings", "ROA", "ROE",
    "Cash_to_Debt", "Working_Capital_Ratio", "Cash from Operating Activity",
    "Borrowing_Growth", "Revenue_Growth", "Profit_Growth"
]

def extract_top_factors(all_shap_df: pd.DataFrame, pred_class: str):
    """
    Extracts only the ESSENTIAL high-impact financial factors (|SHAP| >= 0.04)
    eliminating minor noise line items.
    """
    is_healthy = (pred_class == "Healthy")
    df_ess = all_shap_df[all_shap_df["Feature"].isin(ESSENTIAL_FEATURES)].copy()
    
    risk_list = []
    health_list = []
    
    if is_healthy:
        h_cand = df_ess[df_ess["SHAP_Value"] >= 0.04].sort_values("SHAP_Value", ascending=False).head(4)
        for _, row in h_cand.iterrows():
            feat, val, shap_val = row["Feature"], float(row["Actual_Value"]), float(row["SHAP_Value"])
            health_list.append({
                "feature": feat,
                "name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "actual": format_feature_value(feat, val),
                "impact": f"+{shap_val:.2f} toward Healthy",
                "shap_raw": shap_val,
                "explanation": get_feature_explanation(feat, val, is_risk=False)
            })
            
        r_cand = df_ess[df_ess["SHAP_Value"] <= -0.05].sort_values("SHAP_Value", ascending=True).head(3)
        for _, row in r_cand.iterrows():
            feat, val, shap_val = row["Feature"], float(row["Actual_Value"]), float(row["SHAP_Value"])
            risk_list.append({
                "feature": feat,
                "name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "actual": format_feature_value(feat, val),
                "impact": f"{shap_val:.2f}",
                "shap_raw": shap_val,
                "explanation": get_feature_explanation(feat, val, is_risk=True)
            })
    else:
        r_cand = df_ess[df_ess["SHAP_Value"] >= 0.04].sort_values("SHAP_Value", ascending=False).head(4)
        for _, row in r_cand.iterrows():
            feat, val, shap_val = row["Feature"], float(row["Actual_Value"]), float(row["SHAP_Value"])
            risk_list.append({
                "feature": feat,
                "name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "actual": format_feature_value(feat, val),
                "impact": f"+{shap_val:.2f}",
                "shap_raw": shap_val,
                "explanation": get_feature_explanation(feat, val, is_risk=True)
            })
            
        h_cand = df_ess[df_ess["SHAP_Value"] <= -0.05].sort_values("SHAP_Value", ascending=True).head(2)
        for _, row in h_cand.iterrows():
            feat, val, shap_val = row["Feature"], float(row["Actual_Value"]), float(row["SHAP_Value"])
            health_list.append({
                "feature": feat,
                "name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "actual": format_feature_value(feat, val),
                "impact": f"+{abs(shap_val):.2f} buffer",
                "shap_raw": shap_val,
                "explanation": get_feature_explanation(feat, val, is_risk=False)
            })

    return risk_list, health_list

def generate_dynamic_summary(pred_class: str, conf: float, ratios: dict, top_risk: list, top_health: list) -> str:
    """Generates an executive 2-sentence summary explaining why this company was classified into its predicted class."""
    ic_val = ratios.get("Interest Coverage", "N/A")
    de_val = ratios.get("Debt-to-Equity", "N/A")
    om_val = ratios.get("Operating Margin", "N/A")
    
    if pred_class == "Healthy":
        h_feats = [r["name"] for r in top_health[:2]]
        h_str = " and ".join(h_feats) if h_feats else "strong solvency ratios"
        return (
            f"The model classified this company as <b>Healthy</b> with <b>{conf:.2f}% confidence</b>, "
            f"driven by strong debt servicing coverage ({ic_val}x) and a disciplined capital structure ({de_val}x). "
            f"Favorable operational profitability ({om_val}) combined with robust {h_str} provide a substantial liquidity cushion against corporate financial stress."
        )
    elif pred_class == "Financial Distress":
        r_feats = [r["name"] for r in top_risk[:2]]
        r_str = " and ".join(r_feats) if r_feats else "elevated debt burdens"
        return (
            f"The model classified this company under <b>Financial Distress</b> with <b>{conf:.2f}% confidence</b>, "
            f"reflecting visible compression in debt servicing capacity ({ic_val}x) and rising leverage pressures ({de_val}x). "
            f"Persistent margin headwinds ({om_val}) and vulnerabilities in {r_str} signal tightening liquidity that warrants close credit surveillance."
        )
    elif pred_class == "Probably Bankrupt":
        r_feats = [r["name"] for r in top_risk[:2]]
        r_str = " and ".join(r_feats) if r_feats else "acute debt servicing deficits"
        return (
            f"The model classified this company as <b>Probably Bankrupt</b> with <b>{conf:.2f}% confidence</b>, "
            f"stemming from severe debt servicing strain ({ic_val}x) and extreme balance-sheet leverage ({de_val}x). "
            f"Operating loss generation ({om_val}) and acute stress in {r_str} present heightened default probability under IBC credit stress norms."
        )
    else:  # Bankrupt
        r_feats = [r["name"] for r in top_risk[:2]]
        r_str = " and ".join(r_feats) if r_feats else "complete capital exhaustion"
        return (
            f"The model classified this company as <b>Bankrupt</b> with <b>{conf:.2f}% confidence</b>, "
            f"indicating severe balance sheet insolvency with negative or depleted equity ({de_val}x) and inability to meet interest costs ({ic_val}x). "
            f"Compounded operating losses ({om_val}) and critical deterioration in {r_str} indicate acute default and distressed liquidation risk."
        )

def get_health_snapshot_rows(ratios: dict):
    """Computes status badges and assessment notes for the Financial Health Snapshot table."""
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

    # Assessment logic
    # 1. Interest Coverage
    if ic >= 2.5:
        b_ic = '<span style="background:#DCFCE7; color:#15803D; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #86EFAC;">🟢 Strong</span>'
        n_ic = "Operating profit comfortably exceeds debt finance charges."
    elif ic >= 1.5:
        b_ic = '<span style="background:#FEF9C3; color:#A16207; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FDE047;">🟡 Moderate</span>'
        n_ic = "Adequate coverage but vulnerable to sharp interest rate increases."
    elif ic >= 1.0:
        b_ic = '<span style="background:#FFEDD5; color:#C2410C; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FDBA74;">🟠 Concerning</span>'
        n_ic = "Tight coverage; minimal margin of safety for debt servicing."
    else:
        b_ic = '<span style="background:#FEE2E2; color:#B91C1C; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FCA5A5;">🔴 Critical</span>'
        n_ic = "Operating profit insufficient to service annual interest."

    # 2. Debt to Equity
    if de < 0 or de >= 10.0:
        b_de = '<span style="background:#FEE2E2; color:#B91C1C; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FCA5A5;">🔴 Critical</span>'
        n_de = "Negative equity / eroded net worth; debt severely exceeds capital."
    elif de > 3.0:
        b_de = '<span style="background:#FFEDD5; color:#C2410C; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FDBA74;">🟠 Concerning</span>'
        n_de = "Heavy debt load creates high financial leverage and refinancing risk."
    elif de > 1.5:
        b_de = '<span style="background:#FEF9C3; color:#A16207; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FDE047;">🟡 Moderate</span>'
        n_de = "Moderate leverage within standard industrial peer ranges."
    else:
        b_de = '<span style="background:#DCFCE7; color:#15803D; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #86EFAC;">🟢 Strong</span>'
        n_de = "Conservative leverage provides an ample equity safety buffer."

    # 3. Operating Margin
    if om >= 0.12:
        b_om = '<span style="background:#DCFCE7; color:#15803D; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #86EFAC;">🟢 Strong</span>'
        n_om = "Healthy core business profitability and strong pricing power."
    elif om >= 0.05:
        b_om = '<span style="background:#FEF9C3; color:#A16207; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FDE047;">🟡 Moderate</span>'
        n_om = "Standard operating efficiency across business cycles."
    elif om >= 0.0:
        b_om = '<span style="background:#FFEDD5; color:#C2410C; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FDBA74;">🟠 Concerning</span>'
        n_om = "Thin operating margins vulnerable to inflationary spikes."
    else:
        b_om = '<span style="background:#FEE2E2; color:#B91C1C; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FCA5A5;">🔴 Critical</span>'
        n_om = "Core operations incurring continuous operating losses."

    # 4. ROA
    if roa >= 0.05:
        b_roa = '<span style="background:#DCFCE7; color:#15803D; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #86EFAC;">🟢 Strong</span>'
        n_roa = "High asset productivity and efficient capital deployment."
    elif roa >= 0.01:
        b_roa = '<span style="background:#FEF9C3; color:#A16207; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FDE047;">🟡 Moderate</span>'
        n_roa = "Moderate capital efficiency on total corporate assets."
    elif roa >= 0.0:
        b_roa = '<span style="background:#FFEDD5; color:#C2410C; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FDBA74;">🟠 Concerning</span>'
        n_roa = "Low asset return indicates underperforming capital assets."
    else:
        b_roa = '<span style="background:#FEE2E2; color:#B91C1C; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FCA5A5;">🔴 Critical</span>'
        n_roa = "Total corporate assets generating net negative returns."

    # 5. ROE
    if roe >= 0.12:
        b_roe = '<span style="background:#DCFCE7; color:#15803D; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #86EFAC;">🟢 Strong</span>'
        n_roe = "Strong compounding of equity capital for owners."
    elif roe >= 0.05:
        b_roe = '<span style="background:#FEF9C3; color:#A16207; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FDE047;">🟡 Moderate</span>'
        n_roe = "Reasonable equity compounding rate."
    elif roe >= 0.0:
        b_roe = '<span style="background:#FFEDD5; color:#C2410C; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FDBA74;">🟠 Concerning</span>'
        n_roe = "Subdued equity return."
    else:
        b_roe = '<span style="background:#FEE2E2; color:#B91C1C; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FCA5A5;">🔴 Critical</span>'
        n_roe = "Active erosion of shareholder equity."

    # 6. Working Capital
    if wc >= 0.15:
        b_wc = '<span style="background:#DCFCE7; color:#15803D; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #86EFAC;">🟢 Strong</span>'
        n_wc = "Ample short-term operational liquidity."
    elif wc >= 0.05:
        b_wc = '<span style="background:#FEF9C3; color:#A16207; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FDE047;">🟡 Moderate</span>'
        n_wc = "Adequate working capital to sustain operations."
    elif wc >= 0.0:
        b_wc = '<span style="background:#FFEDD5; color:#C2410C; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FDBA74;">🟠 Concerning</span>'
        n_wc = "Tight working capital requires active working capital loans."
    else:
        b_wc = '<span style="background:#FEE2E2; color:#B91C1C; font-weight:700; padding:3px 10px; border-radius:12px; border:1px solid #FCA5A5;">🔴 Critical</span>'
        n_wc = "Negative working capital; current liabilities exceed current assets."

    rows = [
        ("Interest Coverage Ratio", f"{ic:.2f}x", "≥ 2.50x", b_ic, n_ic),
        ("Debt-to-Equity Ratio", f"{de:.2f}x", "≤ 1.50x", b_de, n_de),
        ("Operating Margin", f"{om:.1%}", "≥ 12.0%", b_om, n_om),
        ("Return on Assets (ROA)", f"{roa:.1%}", "≥ 5.0%", b_roa, n_roa),
        ("Return on Equity (ROE)", f"{roe:.1%}", "≥ 12.0%", b_roe, n_roe),
        ("Working Capital Ratio", f"{wc:.2f}", "≥ 0.15", b_wc, n_wc),
    ]
    return rows

# ---------------------------------------------------------
# 4. Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <div class="header-badge">⚡ Enterprise Solvency Intelligence · 99.46% Accuracy</div>
    <h1 class="header-title">🇮🇳 Corporate Bankruptcy Risk & Explainable AI</h1>
    <p class="header-desc">
        Instant insolvency risk prediction, financial solvency ratios, and Explainable AI (TreeSHAP) in one single dashboard.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. INPUT SECTION (All In One Flow)
# ---------------------------------------------------------
st.markdown("### 1. Select or Enter Company to Test")

mode = st.radio(
    "Choose Analysis Mode:",
    ["🌐 Live Stock Ticker (NSE/BSE)", "🏢 Historical Company (From 999 Dataset)", "✍️ Custom Balance Sheet Numbers"],
    horizontal=True
)

analyzed_data = None

# MODE A: Live Ticker
if mode.startswith("🌐"):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        preset_ticker = st.selectbox(
            "Quick Benchmark Presets:",
            [
                "SUZLON.NS (Suzlon Energy - Clean Energy Turnaround)",
                "IDEA.NS (Vodafone Idea - Heavily Stressed / Telecom)",
                "RELIANCE.NS (Reliance Industries - Conglomerate)",
                "TATASTEEL.NS (Tata Steel - Capital Intensive)",
                "ITC.NS (ITC Ltd - Cash-Rich Blue-Chip)",
                "VEDL.NS (Vedanta Ltd - High Leverage)",
                "YESBANK.NS (Yes Bank - Recovering Financial)",
                "INFY.NS (Infosys - IT Software)",
                "Custom Ticker..."
            ]
        )
    with col2:
        default_sym = preset_ticker.split(" ")[0] if preset_ticker != "Custom Ticker..." else "SUZLON.NS"
        ticker_input = st.text_input("Stock Ticker Symbol (NSE/BSE):", value=default_sym)
    with col3:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Analyze Risk", type="primary")

    if run_btn or "last_live_ticker" not in st.session_state:
        st.session_state["last_live_ticker"] = ticker_input
        with st.spinner(f"Ingesting live exchange filings for {ticker_input}..."):
            try:
                res = fetch_and_predict(ticker_input, active_model=model, explainer=explainer)
                analyzed_data = {
                    "name": f"{res['company_name']} ({res['ticker']})",
                    "subtitle": f"Sector: {res['sector']} | Market Price: ₹{res['current_price']:,.2f} | Latest Filing: {res['period']}",
                    "prediction": res["prediction"],
                    "confidence": res["confidence"],
                    "probabilities": res["probabilities"],
                    "ratios": res["ratios"],
                    "shap_df": res["shap_df"],
                    "all_shap_df": res.get("all_shap_df", res["shap_df"])
                }
                st.session_state["cached_analyzed"] = analyzed_data
            except Exception as e:
                st.error(f"Error fetching live data for '{ticker_input}': {str(e)}")
    else:
        analyzed_data = st.session_state.get("cached_analyzed")

# MODE B: Historical Company Dataset
elif mode.startswith("🏢"):
    if df is not None:
        col1, col2 = st.columns([3, 1])
        company_names = sorted(df["Company"].unique())
        with col1:
            default_idx = company_names.index("ALOK INDUSTRIES LTD") if "ALOK INDUSTRIES LTD" in company_names else 0
            selected_comp = st.selectbox("Search or Select Historical Company:", company_names, index=default_idx)
        with col2:
            st.write("")
            st.write("")
            run_btn_comp = st.button("🔍 Analyze Company", type="primary")

        comp_df = df[df["Company"] == selected_comp].sort_values(by="Year").reset_index(drop=True)
        latest_row = comp_df.iloc[-1]
        
        # Format row into model schema
        row_data = pd.DataFrame([{col: latest_row.get(col, np.nan) for col in features}])
        for col in features:
            row_data[col] = pd.to_numeric(row_data[col], errors="coerce")
        imputed_data = pd.DataFrame(imputer.transform(row_data), columns=features)
        probs = model.predict_proba(imputed_data)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        prob_dict = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(4)}
        
        # Complete SHAP calculation across all 63 features
        sv = explainer(imputed_data)
        vals = sv.values[0, :, pred_idx] if len(sv.values.shape) == 3 else sv.values[0, :]
        
        all_shap_df = pd.DataFrame({
            "Feature": features,
            "SHAP_Value": [float(vals[i]) for i in range(len(features))],
            "Actual_Value": [float(imputed_data.iloc[0, i]) for i in range(len(features))]
        })
        
        top_idx = np.argsort(np.abs(vals))[-10:]
        shap_df = pd.DataFrame({
            "Feature": [features[i] for i in top_idx],
            "SHAP_Value": [float(vals[i]) for i in top_idx],
            "Actual_Value": [float(imputed_data.iloc[0, i]) for i in top_idx]
        })
        
        ic = float(latest_row.get("Interest_Coverage", 0.0))
        de = float(latest_row.get("Debt_to_Equity", 0.0))
        om = float(latest_row.get("Operating_Margin", 0.0))
        roa = float(latest_row.get("ROA", 0.0))
        roe = float(latest_row.get("ROE", 0.0))
        wc = float(latest_row.get("Working_Capital_Ratio", 0.0))
        
        analyzed_data = {
            "name": f"{selected_comp}",
            "subtitle": f"Audited Historical Filing: Fiscal Year {int(latest_row['Year'])}",
            "prediction": pred_class,
            "confidence": prob_dict[pred_class],
            "probabilities": prob_dict,
            "ratios": {
                "Sales (₹ Cr)": f"{latest_row.get('Sales', 0.0):.1f}",
                "Net Profit (₹ Cr)": f"{latest_row.get('Net profit', 0.0):.1f}",
                "Total Debt (₹ Cr)": f"{latest_row.get('Borrowings', 0.0):.1f}",
                "Interest Coverage": f"{ic:.2f}",
                "Debt-to-Equity": f"{de:.2f}",
                "Operating Margin": f"{om:.1%}",
                "Return on Assets (ROA)": f"{roa:.1%}",
                "Return on Equity (ROE)": f"{roe:.1%}",
                "Working Capital Ratio": f"{wc:.2f}"
            },
            "shap_df": shap_df,
            "all_shap_df": all_shap_df
        }

# MODE C: Custom Inputs
else:
    c_preset = st.selectbox("Load Standard Benchmark:", [
        "Healthy Blue-Chip Company",
        "Overleveraged / Stressed Company",
        "High Default Risk (Probably Bankrupt)",
        "Insolvent / Net Worth Eroded (Bankrupt)"
    ])
    defaults = {"Sales": 5000.0, "Net profit": 600.0, "Borrowings": 400.0, "Equity Capital": 100.0, "Reserves": 2500.0, "Debt_to_Equity": 0.15, "Interest_Coverage": 20.0, "Operating_Margin": 0.20}
    if c_preset == "Overleveraged / Stressed Company":
        defaults = {"Sales": 1800.0, "Net profit": 30.0, "Borrowings": 1800.0, "Equity Capital": 80.0, "Reserves": 400.0, "Debt_to_Equity": 3.75, "Interest_Coverage": 1.4, "Operating_Margin": 0.04}
    elif c_preset == "High Default Risk (Probably Bankrupt)":
        defaults = {"Sales": 900.0, "Net profit": -150.0, "Borrowings": 2500.0, "Equity Capital": 50.0, "Reserves": 100.0, "Debt_to_Equity": 8.0, "Interest_Coverage": 0.3, "Operating_Margin": -0.08}
    elif c_preset == "Insolvent / Net Worth Eroded (Bankrupt)":
        defaults = {"Sales": 400.0, "Net profit": -600.0, "Borrowings": 3500.0, "Equity Capital": 50.0, "Reserves": -800.0, "Debt_to_Equity": -4.6, "Interest_Coverage": -2.0, "Operating_Margin": -0.30}

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        s_sales = st.number_input("Sales (₹ Cr)", value=float(defaults["Sales"]))
        s_profit = st.number_input("Net Profit (₹ Cr)", value=float(defaults["Net profit"]))
    with r2:
        s_debt = st.number_input("Borrowings (₹ Cr)", value=float(defaults["Borrowings"]))
        s_eq = st.number_input("Equity (₹ Cr)", value=float(defaults["Equity Capital"]) + float(defaults["Reserves"]))
    with r3:
        s_ic = st.number_input("Interest Coverage (EBIT/Interest)", value=float(defaults["Interest_Coverage"]))
        s_de = st.number_input("Debt-to-Equity Ratio", value=float(defaults["Debt_to_Equity"]))
    with r4:
        s_om = st.number_input("Operating Margin", value=float(defaults["Operating_Margin"]))
        st.write("")
        st.write("")
        custom_btn = st.button("🧪 Predict Custom Risk", type="primary")

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
    c_pred = CLASS_NAMES[c_idx]
    c_prob_dict = {CLASS_NAMES[i]: round(float(c_probs[i]) * 100, 2) for i in range(4)}
    
    c_sv = explainer(c_imp)
    c_vals = c_sv.values[0, :, c_idx] if len(c_sv.values.shape) == 3 else c_sv.values[0, :]
    
    c_all_shap_df = pd.DataFrame({
        "Feature": features,
        "SHAP_Value": [float(c_vals[i]) for i in range(len(features))],
        "Actual_Value": [float(c_imp.iloc[0, i]) for i in range(len(features))]
    })
    
    c_top_idx = np.argsort(np.abs(c_vals))[-10:]
    c_shap_df = pd.DataFrame({
        "Feature": [features[i] for i in c_top_idx],
        "SHAP_Value": [float(c_vals[i]) for i in c_top_idx],
        "Actual_Value": [float(c_imp.iloc[0, i]) for i in c_top_idx]
    })
    
    analyzed_data = {
        "name": f"Custom Simulation: {c_preset}",
        "subtitle": "Hypothetical Balance Sheet & Solvency Numbers",
        "prediction": c_pred,
        "confidence": c_prob_dict[c_pred],
        "probabilities": c_prob_dict,
        "ratios": {
            "Sales (₹ Cr)": f"{s_sales:.1f}",
            "Net Profit (₹ Cr)": f"{s_profit:.1f}",
            "Total Debt (₹ Cr)": f"{s_debt:.1f}",
            "Interest Coverage": f"{s_ic:.2f}",
            "Debt-to-Equity": f"{s_de:.2f}",
            "Operating Margin": f"{s_om:.1%}",
            "Return on Assets (ROA)": f"{float(c_imp.iloc[0]['ROA']):.1%}",
            "Return on Equity (ROE)": f"{float(c_imp.iloc[0]['ROE']):.1%}",
            "Working Capital Ratio": f"{float(c_imp.iloc[0]['Working_Capital_Ratio']):.2f}"
        },
        "shap_df": c_shap_df,
        "all_shap_df": c_all_shap_df
    }

# ---------------------------------------------------------
# 6. UNIFIED COLORFUL OUTPUT DASHBOARD
# ---------------------------------------------------------
if analyzed_data is not None:
    st.divider()
    
    pred = analyzed_data["prediction"]
    conf = analyzed_data["confidence"]
    all_shap_df = analyzed_data["all_shap_df"]
    ratios = analyzed_data["ratios"]
    
    # Extract dynamic Top 5 Risk and Top 3 Health factors
    risk_factors, health_factors = extract_top_factors(all_shap_df, pred)
    
    # 1. Company Name & Subtitle
    st.markdown(f"## 🏢 **{analyzed_data['name']}**")
    st.caption(analyzed_data["subtitle"])
    
    # 2. Rich Colorful Verdict Card
    box_class = {
        "Healthy": "verdict-healthy",
        "Financial Distress": "verdict-distress",
        "Probably Bankrupt": "verdict-prob-bankrupt",
        "Bankrupt": "verdict-bankrupt"
    }[pred]
    
    color = CLASS_COLORS[pred]
    
    badge_label = {
        "Healthy": "🟢 HEALTHY / SOLVENT",
        "Financial Distress": "🟡 FINANCIAL DISTRESS",
        "Probably Bankrupt": "🟠 PROBABLY BANKRUPT",
        "Bankrupt": "🔴 BANKRUPT / INSOLVENT"
    }[pred]

    st.markdown(f"""
    <div class="verdict-box {box_class}">
        <div style="font-size: 0.95rem; font-weight: 700; color: #475569; letter-spacing: 0.08em; text-transform: uppercase;">
            AI Solvency & Insolvency Verdict
        </div>
        <div style="font-size: 2.3rem; font-weight: 900; color: {color}; margin: 8px 0;">
            {badge_label}
        </div>
        <div style="display: inline-block; background: rgba(0,0,0,0.06); padding: 5px 16px; border-radius: 20px; font-size: 1.05rem; font-weight: 700; color: #1E293B;">
            Model Confidence: <span style="color: {color};">{conf:.2f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Dynamic Executive Summary (2-Sentence Plain English)
    summary_text = generate_dynamic_summary(pred, conf, ratios, risk_factors, health_factors)
    st.markdown(f"""
    <div class="summary-box">
        <div style="font-size: 0.85rem; font-weight: 800; color: #1D4ED8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
            💡 Executive Decision Rationale
        </div>
        <div style="font-size: 1.02rem; color: #1E293B; line-height: 1.6;">
            {summary_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4. Multi-Class Risk Distribution & Key Financial Indicators
    col_dist, col_metrics = st.columns([1, 1])
    
    with col_dist:
        st.markdown("#### 🎯 Multi-Class Probability Distribution")
        fig_prob = px.bar(
            x=list(analyzed_data["probabilities"].keys()),
            y=list(analyzed_data["probabilities"].values()),
            color=list(analyzed_data["probabilities"].keys()),
            color_discrete_map=CLASS_COLORS,
            labels={"x": "Risk Category", "y": "Probability (%)"},
            text=[f"{v:.1f}%" for v in analyzed_data["probabilities"].values()]
        )
        fig_prob.update_layout(
            yaxis=dict(range=[0, 105], title="Model Probability (%)"),
            xaxis=dict(title=""),
            showlegend=False,
            height=280,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
    with col_metrics:
        st.markdown("#### 📊 Key Financial Indicators")
        # Display 6 key metrics in a 3x2 colorful card grid
        m_items = [
            ("Sales Revenue", ratios.get("Sales (₹ Cr)", "N/A"), "#EFF6FF", "#3B82F6"),
            ("Net Profit", ratios.get("Net Profit (₹ Cr)", "N/A"), "#F0FDF4" if not str(ratios.get("Net Profit (₹ Cr)", "")).startswith("-") else "#FEF2F2", "#10B981" if not str(ratios.get("Net Profit (₹ Cr)", "")).startswith("-") else "#EF4444"),
            ("Total Debt", ratios.get("Total Debt (₹ Cr)", "N/A"), "#FAF5FF", "#8B5CF6"),
            ("Interest Coverage", f"{ratios.get('Interest Coverage', 'N/A')}x", "#ECFDF5" if float(str(ratios.get('Interest Coverage', 0)).replace('x','')) >= 2.0 else "#FEF2F2", "#10B981" if float(str(ratios.get('Interest Coverage', 0)).replace('x','')) >= 2.0 else "#EF4444"),
            ("Debt-to-Equity", f"{ratios.get('Debt-to-Equity', 'N/A')}x", "#ECFDF5" if float(str(ratios.get('Debt-to-Equity', 0)).replace('x','')) <= 1.5 and float(str(ratios.get('Debt-to-Equity', 0)).replace('x','')) > 0 else "#FEF2F2", "#10B981" if float(str(ratios.get('Debt-to-Equity', 0)).replace('x','')) <= 1.5 and float(str(ratios.get('Debt-to-Equity', 0)).replace('x','')) > 0 else "#EF4444"),
            ("Operating Margin", str(ratios.get("Operating Margin", "N/A")), "#ECFDF5" if not str(ratios.get("Operating Margin", "")).startswith("-") else "#FEF2F2", "#10B981" if not str(ratios.get("Operating Margin", "")).startswith("-") else "#EF4444")
        ]
        
        row1 = st.columns(3)
        row2 = st.columns(3)
        for i, (label, val, bg, border_c) in enumerate(m_items[:3]):
            with row1[i]:
                st.markdown(f"""
                <div class="metric-card" style="background: {bg}; border-left: 4px solid {border_c};">
                    <div class="metric-label">{label}</div>
                    <div class="metric-val">{val}</div>
                </div>
                """, unsafe_allow_html=True)
        for i, (label, val, bg, border_c) in enumerate(m_items[3:6]):
            with row2[i]:
                st.markdown(f"""
                <div class="metric-card" style="background: {bg}; border-left: 4px solid {border_c};">
                    <div class="metric-label">{label}</div>
                    <div class="metric-val">{val}</div>
                </div>
                """, unsafe_allow_html=True)

    st.write("")

    # ---------------------------------------------------------
    # 7. SECTION: WHY DID THE MODEL MAKE THIS PREDICTION?
    # ---------------------------------------------------------
    st.markdown("### 🔍 Why did the model make this prediction?")
    st.caption("Detailed breakdown of the strongest individual factors influencing the model's prediction, derived from real-time TreeSHAP contributions and audited financial values.")

    col_risk, col_health = st.columns([1, 1])

    # RED COLUMN: Top 5 Factors Increasing Financial Risk
    with col_risk:
        st.markdown("""
        <div class="factor-container-risk">
            <div style="font-size: 1.15rem; font-weight: 800; color: #BE123C; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between;">
                <span>🔴 Factors Increasing Financial Risk</span>
                <span style="background: #FFE4E6; color: #9F1239; font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 12px;">TOP 5 RISK DRIVERS</span>
            </div>
        """, unsafe_allow_html=True)
        
        if not risk_factors:
            st.markdown(
                '<div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:10px; padding:14px; color:#166534; font-size:0.9rem;">'
                '<b>✅ No Material Solvency Risks Detected</b><br>'
                'All primary debt servicing, leverage, and operating margins are operating comfortably within safe regulatory zones.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            for item in risk_factors:
                card_html = (
                    f'<div class="factor-item-card" style="border-left: 4px solid #E11D48;">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
                    f'<span style="font-weight: 800; color: #0F172A; font-size: 1.02rem;">{item["name"]}</span>'
                    f'<span style="background: #FEE2E2; color: #991B1B; font-weight: 700; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; border: 1px solid #FECDD3;">Impact: {item["impact"]}</span>'
                    f'</div>'
                    f'<div style="font-size: 0.88rem; color: #64748B; margin-bottom: 6px;"><b>Reported Value:</b> <span style="color: #991B1B; font-weight: 700;">{item["actual"]}</span></div>'
                    f'<div style="font-size: 0.92rem; color: #334155; line-height: 1.45;"><b>Reason:</b> {item["explanation"]}</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # GREEN COLUMN: Factors Supporting Financial Health
    with col_health:
        st.markdown("""
<div class="factor-container-health">
    <div style="font-size: 1.15rem; font-weight: 800; color: #15803D; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between;">
        <span>🟢 Factors Supporting Financial Health</span>
        <span style="background: #DCFCE7; color: #166534; font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 12px;">KEY HEALTH PILLARS</span>
    </div>
""", unsafe_allow_html=True)
        
        if not health_factors:
            st.markdown(
                '<div style="background:#FFF7ED; border:1px solid #FDBA74; border-radius:10px; padding:14px; color:#9A3412; font-size:0.9rem;">'
                '<b>⚠️ High Default Vulnerability</b><br>'
                'No substantial solvency buffers present across core debt servicing or balance sheet metrics.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            for item in health_factors:
                card_html = (
                    f'<div class="factor-item-card" style="border-left: 4px solid #16A34A;">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
                    f'<span style="font-weight: 800; color: #0F172A; font-size: 1.02rem;">{item["name"]}</span>'
                    f'<span style="background: #DCFCE7; color: #14532D; font-weight: 700; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; border: 1px solid #BBF7D0;">Impact: {item["impact"]}</span>'
                    f'</div>'
                    f'<div style="font-size: 0.88rem; color: #64748B; margin-bottom: 6px;"><b>Reported Value:</b> <span style="color: #166534; font-weight: 700;">{item["actual"]}</span></div>'
                    f'<div style="font-size: 0.92rem; color: #334155; line-height: 1.45;"><b>Reason:</b> {item["explanation"]}</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ---------------------------------------------------------
    # 9. SECTION: DUAL-COLORED SHAP FACTOR IMPACT CHART
    # ---------------------------------------------------------
    st.markdown("### 🧠 SHAP Factor Attribution Chart")
    st.caption("Exact signed mathematical SHAP values. Green = pushes toward Solvency/Health; Red = pushes toward Financial Distress/Bankruptcy.")
    
    col_chart, col_tbl = st.columns([3, 2])
    
    with col_chart:
        shap_df = analyzed_data["shap_df"]
        is_healthy = (pred == "Healthy")
        
        # Friendly labels for chart
        chart_features = [FEATURE_DISPLAY_NAMES.get(f, f.replace("_", " ")) for f in shap_df["Feature"]]
        
        # Color: Emerald green for health contributors, Crimson red for risk contributors
        bar_colors = [
            "#10B981" if ((v > 0 and is_healthy) or (v < 0 and not is_healthy)) else "#EF4444" 
            for v in shap_df["SHAP_Value"]
        ]
        
        fig_shap = go.Figure(go.Bar(
            x=shap_df["SHAP_Value"],
            y=chart_features,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="rgba(0,0,0,0.1)", width=1)
            ),
            text=[f"{v:+.3f} (Val: {format_feature_value(f, act)})" for f, v, act in zip(shap_df["Feature"], shap_df["SHAP_Value"], shap_df["Actual_Value"])],
            textposition="auto"
        ))
        fig_shap.update_layout(
            title=dict(
                text=f"SHAP Factor Contributions on '{pred}' Verdict",
                font=dict(size=14, color="#1E293B")
            ),
            xaxis_title="SHAP Value (Marginal Impact on Log-Odds)",
            yaxis_title="",
            height=340,
            margin=dict(l=10, r=10, t=35, b=10),
            plot_bgcolor="rgba(248, 250, 252, 0.5)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        
    with col_tbl:
        st.markdown("#### 📋 Top Factor Data Table")
        disp_df = shap_df.iloc[::-1].copy()
        disp_df["Friendly Name"] = [FEATURE_DISPLAY_NAMES.get(f, f.replace("_", " ")) for f in disp_df["Feature"]]
        disp_df["Reported Value"] = [format_feature_value(f, val) for f, val in zip(disp_df["Feature"], disp_df["Actual_Value"])]
        disp_df["SHAP Impact"] = [f"{v:+.4f}" for v in disp_df["SHAP_Value"]]
        
        st.dataframe(
            disp_df[["Friendly Name", "Reported Value", "SHAP Impact"]].rename(columns={"Friendly Name": "Feature"}),
            use_container_width=True,
            hide_index=True
        )
