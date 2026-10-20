import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap

from live_data_fetcher import fetch_and_predict, normalize_ticker

# 1. Page Configuration
st.set_page_config(
    page_title="Corporate Bankruptcy Prediction & Explainable AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"  # Keep sidebar collapsed for clean, distraction-free UI
)

# 2. Clean Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
        text-align: center;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .verdict-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #E2E8F0;
    }
    .stButton>button {
        width: 100%;
        height: 48px;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Directory Setup & Asset Loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "Indian Company Financial Dataset.csv")

CLASS_NAMES = ["Bankrupt", "Financial Distress", "Probably Bankrupt", "Healthy"]
CLASS_COLORS = {
    "Healthy": "#16A34A",
    "Financial Distress": "#CA8A04",
    "Probably Bankrupt": "#EA580C",
    "Bankrupt": "#DC2626"
}
CLASS_BG = {
    "Healthy": "#F0FDF4",
    "Financial Distress": "#FEFCE8",
    "Probably Bankrupt": "#FFF7ED",
    "Bankrupt": "#FEF2F2"
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
    st.image("https://img.icons8.com/color/96/bank-building.png", width=50)
    st.markdown("### 🤖 Model Info")
    st.markdown("""
    * **Algorithm:** CatBoost Classifier
    * **Verified Accuracy:** **99.46%**
    * **Macro F1 Score:** 0.9934
    * **Training Data:** 8,991 Audited Filings (999 Indian Firms)
    * **Explainability:** Real-Time TreeSHAP
    """)
    st.divider()
    st.caption("Antigravity AI · Unified Solvency Engine")

# Header
st.markdown('<div class="main-title">🇮🇳 Corporate Bankruptcy Risk & Explainable AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Instant insolvency risk prediction, financial solvency ratios, and Explainable AI (SHAP) in one single dashboard.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# INPUT SECTION (All In One Simple Flow)
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
                    "shap_df": res["shap_df"]
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
        
        # SHAP calculation
        sv = explainer(imputed_data)
        vals = sv.values[0, :, pred_idx] if len(sv.values.shape) == 3 else sv.values[0, :]
        top_idx = np.argsort(np.abs(vals))[-8:]
        shap_df = pd.DataFrame({
            "Feature": [features[i] for i in top_idx],
            "SHAP_Value": [float(vals[i]) for i in top_idx],
            "Actual_Value": [float(imputed_data.iloc[0, i]) for i in top_idx]
        })
        
        ic = float(latest_row.get("Interest_Coverage", 0.0))
        de = float(latest_row.get("Debt_to_Equity", 0.0))
        om = float(latest_row.get("Operating_Margin", 0.0))
        
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
                "Operating Margin": f"{om:.1%}"
            },
            "shap_df": shap_df
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
    c_top_idx = np.argsort(np.abs(c_vals))[-8:]
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
            "Operating Margin": f"{s_om:.1%}"
        },
        "shap_df": c_shap_df
    }

# ---------------------------------------------------------
# UNIFIED OUTPUT RESULTS DASHBOARD (Everything In One Place)
# ---------------------------------------------------------
if analyzed_data is not None:
    st.divider()
    
    # 1. Company / Simulation Header
    st.markdown(f"## 🏢 **{analyzed_data['name']}**")
    st.caption(analyzed_data["subtitle"])
    
    # 2. Risk Verdict Banner
    pred = analyzed_data["prediction"]
    conf = analyzed_data["confidence"]
    color = CLASS_COLORS[pred]
    bg_color = CLASS_BG[pred]
    
    st.markdown(f"""
    <div class="verdict-card" style="background-color: {bg_color}; border-left: 8px solid {color};">
        <div style="font-size: 1.1rem; color: #475569; font-weight: 600;">AI SOLVENCY & BANKRUPTCY CLASSIFICATION</div>
        <div style="font-size: 2.5rem; font-weight: 800; color: {color}; margin-top: 4px;">{pred.upper()}</div>
        <div style="font-size: 1.15rem; font-weight: 600; color: #1E293B;">Confidence Score: {conf:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Multi-Class Probability Breakdown & Key Ratios
    col_out1, col_out2 = st.columns([1, 1])
    
    with col_out1:
        st.markdown("#### 🎯 Risk Probability Distribution")
        fig_prob = px.bar(
            x=list(analyzed_data["probabilities"].keys()),
            y=list(analyzed_data["probabilities"].values()),
            color=list(analyzed_data["probabilities"].keys()),
            color_discrete_map=CLASS_COLORS,
            labels={"x": "Risk Category", "y": "Probability (%)"},
            text=[f"{v:.1f}%" for v in analyzed_data["probabilities"].values()]
        )
        fig_prob.update_layout(yaxis=dict(range=[0, 105]), showlegend=False, height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_prob, use_container_width=True)
        
    with col_out2:
        st.markdown("#### 📊 Key Solvency Indicators")
        # Display ratios in clean 3x2 grid
        r_items = list(analyzed_data["ratios"].items())
        r_cols1 = st.columns(3)
        r_cols2 = st.columns(3)
        for i, (k, v) in enumerate(r_items[:3]):
            with r_cols1[i]:
                st.metric(k, str(v))
        for i, (k, v) in enumerate(r_items[3:6]):
            with r_cols2[i]:
                st.metric(k, str(v))
                
        # Simple Solvency Rule Alert
        if pred in ["Bankrupt", "Probably Bankrupt", "Financial Distress"]:
            st.error("⚠️ **Risk Alert:** Low/negative interest coverage or excessive debt-to-equity ratio signals acute debt servicing strain under IBC/RBI stress norms.")
        else:
            st.success("✅ **Solvency Passed:** Comfortable earnings coverage and manageable leverage indicate healthy ongoing solvency.")

    st.divider()

    # 4. EXPLAINABLE AI (SHAP) - INTEGRATED DIRECTLY BELOW
    st.markdown("### 🧠 Explainable AI (SHAP) - Why did the AI make this decision?")
    st.caption("Exact mathematical feature contributions computed via TreeSHAP. Red = pushed towards Distress/Bankruptcy; Green = pushed towards Solvency/Health.")
    
    shap_c1, shap_c2 = st.columns([3, 2])
    
    with shap_c1:
        shap_df = analyzed_data["shap_df"]
        is_healthy = (pred == "Healthy")
        bar_colors = ["#16A34A" if ((v > 0 and is_healthy) or (v < 0 and not is_healthy)) else "#DC2626" for v in shap_df["SHAP_Value"]]
        
        fig_shap = go.Figure(go.Bar(
            x=shap_df["SHAP_Value"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.3f} (Val: {act:.2f})" for v, act in zip(shap_df["SHAP_Value"], shap_df["Actual_Value"])],
            textposition="auto"
        ))
        fig_shap.update_layout(
            title=f"SHAP Factor Impacts on '{pred}' Verdict",
            xaxis_title="SHAP Value (Marginal Impact on Probability)",
            yaxis_title="Financial Ratio",
            height=320,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        
    with shap_c2:
        st.markdown("#### 📋 Top Factor Table")
        st.dataframe(shap_df.iloc[::-1].rename(columns={
            "SHAP_Value": "SHAP Impact",
            "Actual_Value": "Reported Value"
        }).style.format({
            "SHAP Impact": "{:+.4f}",
            "Reported Value": "{:.2f}"
        }), use_container_width=True)