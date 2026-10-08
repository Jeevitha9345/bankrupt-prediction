import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASS_NAMES = ["Bankrupt", "Financial Distress", "Probably Bankrupt", "Healthy"]

def load_resources():
    features = joblib.load(os.path.join(MODELS_DIR, "indian_feature_list.pkl"))
    imputer = joblib.load(os.path.join(MODELS_DIR, "indian_imputer.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "catboost_indian.pkl"))
    return features, imputer, model

def normalize_ticker(symbol: str) -> str:
    symbol = symbol.strip().upper()
    # If user provided a pure Indian ticker symbol without exchange suffix, default to NSE (.NS)
    if "." not in symbol and not symbol.endswith((".NS", ".BO")):
        # If it looks like a numeric BSE code
        if symbol.isdigit():
            return f"{symbol}.BO"
        return f"{symbol}.NS"
    return symbol

def _get_metric_val(df: pd.DataFrame, candidate_names: list, default=0.0, scale=1e7) -> float:
    """Helper to safely extract the most recent valid financial metric from yfinance DataFrames."""
    if df is None or df.empty:
        return default
    for name in candidate_names:
        if name in df.index:
            series = df.loc[name].dropna()
            if not series.empty:
                val = series.iloc[0]
                if pd.notnull(val):
                    return float(val) / scale
    return default

def _get_growth_rate(df: pd.DataFrame, candidate_names: list, default=0.0) -> float:
    """Helper to safely calculate 1-year growth rate from the latest two periods."""
    if df is None or df.empty:
        return default
    for name in candidate_names:
        if name in df.index:
            series = df.loc[name].dropna()
            if len(series) >= 2:
                v_curr = float(series.iloc[0])
                v_prev = float(series.iloc[1])
                if v_prev != 0:
                    return (v_curr - v_prev) / abs(v_prev)
    return default

def fetch_and_predict(ticker_input: str, active_model=None, explainer=None):
    """
    Fetches live financial statements for any stock ticker from Yahoo Finance,
    transforms them into the 63-feature model schema, predicts 4-tier risk status,
    and returns real-time SHAP factor attributions.
    """
    features, imputer, default_model = load_resources()
    model = active_model if active_model is not None else default_model

    ticker_sym = normalize_ticker(ticker_input)
    t = yf.Ticker(ticker_sym)

    # 1. Company Metadata
    try:
        info = t.info
    except Exception:
        info = {}

    company_name = info.get("shortName") or info.get("longName") or ticker_sym
    sector = info.get("sector") or "N/A"
    industry = info.get("industry") or "N/A"
    market_cap_raw = info.get("marketCap", 0)
    market_cap_cr = (market_cap_raw / 1e7) if market_cap_raw else 0.0
    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
    currency = info.get("currency") or "INR"

    # 2. Financial Statements (Annual preferred, fallback to quarterly)
    bs = t.balance_sheet if (t.balance_sheet is not None and not t.balance_sheet.empty) else t.quarterly_balance_sheet
    fin = t.financials if (t.financials is not None and not t.financials.empty) else t.quarterly_financials
    cf = t.cashflow if (t.cashflow is not None and not t.cashflow.empty) else t.quarterly_cashflow

    if bs is None or bs.empty or fin is None or fin.empty:
        raise ValueError(f"Could not retrieve complete financial statements for ticker '{ticker_sym}'. Please verify the symbol.")

    latest_period = str(fin.columns[0]).split(" ")[0] if not fin.empty else "Latest Available"

    # 3. Extract Fundamental Line Items (Values in INR Crores / Millions for non-INR)
    sales = _get_metric_val(fin, ["Total Revenue", "Operating Revenue", "Gross Revenue"], default=100.0)
    net_profit = _get_metric_val(fin, ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"], default=5.0)
    operating_profit = _get_metric_val(fin, ["Operating Income", "EBIT", "Normalized EBITDA"], default=10.0)
    interest = abs(_get_metric_val(fin, ["Interest Expense", "Interest Expense Non Operating", "Net Interest Income"], default=2.0))
    depreciation = abs(_get_metric_val(fin, ["Reconciled Depreciation", "Depreciation And Amortization In Income Statement", "Depreciation Amortization Depletion"], default=5.0))
    pbt = _get_metric_val(fin, ["Pretax Income"], default=net_profit * 1.25)
    tax = _get_metric_val(fin, ["Tax Provision"], default=abs(pbt - net_profit) if pbt > net_profit else 0.0)

    borrowings = _get_metric_val(bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"], default=50.0)
    equity = _get_metric_val(bs, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"], default=50.0)
    assets = _get_metric_val(bs, ["Total Assets"], default=150.0)
    liabilities = _get_metric_val(bs, ["Total Liabilities Net Minority Interest", "Total Liabilities"], default=assets - equity)
    cash = _get_metric_val(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash Financial"], default=10.0)
    receivables = _get_metric_val(bs, ["Receivables", "Accounts Receivable"], default=15.0)
    inventory = _get_metric_val(bs, ["Inventory"], default=10.0)
    working_capital = _get_metric_val(bs, ["Working Capital"], default=assets * 0.1)

    ocf = _get_metric_val(cf, ["Operating Cash Flow", "Cash Flowsfromusedin Operating Activities"], default=operating_profit * 0.8)
    icf = _get_metric_val(cf, ["Investing Cash Flow", "Cash Flowsfromusedin Investing Activities"], default=-10.0)
    fcf = _get_metric_val(cf, ["Financing Cash Flow", "Cash Flowsfromusedin Financing Activities"], default=-5.0)

    # 4. Compute Financial Ratios
    interest_coverage = (operating_profit / interest) if interest > 0 else (5.0 if operating_profit > 0 else -1.0)
    debt_to_equity = (borrowings / equity) if equity > 0 else (10.0 if borrowings > 0 else -5.0)
    operating_margin = (operating_profit / sales) if sales > 0 else 0.05
    profit_margin = (net_profit / sales) if sales > 0 else 0.05
    roa = (net_profit / assets) if assets > 0 else 0.05
    roe = (net_profit / equity) if equity > 0 else 0.05
    working_capital_ratio = (working_capital / assets) if assets > 0 else 0.05
    asset_utilization = (sales / assets) if assets > 0 else 0.5
    receivable_ratio = (receivables / sales) if sales > 0 else 0.1
    inventory_ratio = (inventory / sales) if sales > 0 else 0.1
    cash_to_debt = (cash / borrowings) if borrowings > 0 else 1.0
    cash_margin = (ocf / sales) if sales > 0 else 0.05
    expense_ratio = ((sales - operating_profit) / sales) if sales > 0 else 0.8
    depreciation_ratio = (depreciation / assets) if assets > 0 else 0.03

    # Multi-Year Growth Ratios
    rev_growth = _get_growth_rate(fin, ["Total Revenue", "Operating Revenue"])
    prof_growth = _get_growth_rate(fin, ["Net Income", "Net Income Common Stockholders"])
    debt_growth = _get_growth_rate(bs, ["Total Debt"])
    asset_growth = _get_growth_rate(bs, ["Total Assets"])

    # 5. Populate Complete 63-Feature Vector
    raw_dict = {f: np.nan for f in features}
    raw_dict.update({
        "Sales": sales,
        "Net profit": net_profit,
        "Operating Profit": operating_profit,
        "Interest": interest,
        "Depreciation": depreciation,
        "Profit before tax": pbt,
        "Tax": tax,
        "Borrowings": borrowings,
        "Equity Share Capital": equity * 0.25 if equity > 0 else 10.0,
        "Reserves": equity * 0.75 if equity > 0 else equity,
        "Total Assets": assets,
        "Total Liabilities": liabilities,
        "Cash & Bank": cash,
        "Receivables": receivables,
        "Inventory": inventory,
        "Cash from Operating Activity": ocf,
        "Cash from Investing Activity": icf,
        "Cash from Financing Activity": fcf,
        "Net Cash Flow": ocf + icf + fcf,
        "Interest_Coverage": interest_coverage,
        "Debt_to_Equity": debt_to_equity,
        "Operating_Margin": operating_margin,
        "Profit_Margin": profit_margin,
        "ROA": roa,
        "ROE": roe,
        "Working_Capital_Ratio": working_capital_ratio,
        "Asset_Utilization": asset_utilization,
        "Receivable_Ratio": receivable_ratio,
        "Inventory_Ratio": inventory_ratio,
        "Cash_to_Debt": cash_to_debt,
        "Cash_Margin": cash_margin,
        "Expense_Ratio": expense_ratio,
        "Depreciation_Ratio": depreciation_ratio,
        "Revenue_Growth": rev_growth,
        "Profit_Growth": prof_growth,
        "Borrowing_Growth": debt_growth,
        "Asset_Growth": asset_growth,
    })

    input_df = pd.DataFrame([raw_dict])
    imputed_df = pd.DataFrame(imputer.transform(input_df), columns=features)

    # 6. Model Prediction
    probs = model.predict_proba(imputed_df)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    prob_dict = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(4)}

    # 7. Local SHAP Attribution
    if explainer is None:
        explainer = shap.TreeExplainer(model)
        
    sv = explainer(imputed_df)
    if len(sv.values.shape) == 3:
        if sv.values.shape[0] == 1:
            vals = sv.values[0, :, pred_idx]
        else:
            vals = sv.values[pred_idx, 0, :]
    else:
        vals = sv.values[0, :]

    top_idx = np.argsort(np.abs(vals))[-10:]
    shap_features = [features[i] for i in top_idx]
    shap_values = [float(vals[i]) for i in top_idx]
    actual_values = [float(imputed_df.iloc[0, i]) for i in top_idx]

    shap_df = pd.DataFrame({
        "Feature": shap_features,
        "SHAP_Value": shap_values,
        "Actual_Value": actual_values
    })

    return {
        "ticker": ticker_sym,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "currency": currency,
        "market_cap_cr": market_cap_cr,
        "current_price": current_price,
        "period": latest_period,
        "prediction": pred_class,
        "confidence": prob_dict[pred_class],
        "probabilities": prob_dict,
        "shap_df": shap_df,
        "ratios": {
            "Sales (₹ Cr)": round(sales, 1),
            "Net Profit (₹ Cr)": round(net_profit, 1),
            "Operating Profit (₹ Cr)": round(operating_profit, 1),
            "Total Debt (₹ Cr)": round(borrowings, 1),
            "Total Equity (₹ Cr)": round(equity, 1),
            "Interest Coverage": round(interest_coverage, 2),
            "Debt-to-Equity": round(debt_to_equity, 2),
            "Operating Margin": f"{operating_margin:.1%}",
            "Return on Equity (ROE)": f"{roe:.1%}",
            "Return on Assets (ROA)": f"{roa:.1%}",
            "Working Capital Ratio": f"{working_capital_ratio:.2f}",
        }
    }
