import argparse
import sys
from live_data_fetcher import fetch_and_predict

PRESET_TICKERS = {
    "1": ("RELIANCE.NS", "Reliance Industries (Blue-Chip Conglomerate)"),
    "2": ("SUZLON.NS", "Suzlon Energy (Turnaround / Restructured Clean Energy)"),
    "3": ("IDEA.NS", "Vodafone Idea (Heavily Leveraged / Stressed Telecom)"),
    "4": ("TATASTEEL.NS", "Tata Steel (Capital-Intensive Manufacturing)"),
    "5": ("ITC.NS", "ITC Ltd (Cash-Rich FMCG Blue-Chip)"),
    "6": ("VEDL.NS", "Vedanta Ltd (Metals & Mining)"),
    "7": ("YESBANK.NS", "Yes Bank (Recovering Financial)"),
    "8": ("BHEL.NS", "Bharat Heavy Electricals (Public Sector Industrial)")
}

def display_assessment(res):
    print("=" * 70)
    print(f"       LIVE REAL-TIME CORPORATE RISK ASSESSMENT")
    print("=" * 70)
    print(f"Company Name : {res['company_name']}")
    print(f"Ticker Symbol: {res['ticker']}")
    print(f"Sector / Ind : {res['sector']} | {res['industry']}")
    print(f"Market Price : Rs. {res['current_price']:,.2f} {res['currency']}")
    print(f"Market Cap   : Rs. {res['market_cap_cr']:,.1f} Cr")
    print(f"Latest Filing: {res['period']}")
    print("-" * 70)
    
    # Verdict
    pred = res['prediction']
    conf = res['confidence']
    badge = "[SAFE]" if pred == "Healthy" else "[DISTRESS]" if pred == "Financial Distress" else "[HIGH RISK]" if pred == "Probably Bankrupt" else "[DEFAULT/BANKRUPT]"
    print(f"AI RISK VERDICT : {badge} {pred.upper()} ({conf:.2f}% Confidence)")
    print("-" * 70)
    print("MULTI-CLASS PROBABILITIES:")
    for cname, pval in res['probabilities'].items():
        bar = "#" * int(pval / 5)
        print(f"  {cname:<20}: {pval:>6.2f}% | {bar}")
        
    print("-" * 70)
    print("KEY FINANCIAL SOLVENCY RATIOS:")
    for k, v in res['ratios'].items():
        clean_k = k.replace("₹", "Rs.")
        print(f"  * {clean_k:<25}: {v}")
        
    print("-" * 70)
    print("EXPLAINABLE AI (SHAP) - TOP MATHEMATICAL DRIVERS:")
    shap_df = res['shap_df']
    for _, row in shap_df.iloc[::-1].head(6).iterrows():
        feat = row['Feature']
        s_val = row['SHAP_Value']
        act = row['Actual_Value']
        impact_dir = "+ Pushes Toward " + pred if s_val > 0 else "- Pulls Away From " + pred
        print(f"  * {feat:<22}: Reported = {act:>10.2f} | SHAP = {s_val:>+7.4f} ({impact_dir})")
        
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Test Live Indian Stock Ticker against Corporate Bankruptcy Model.")
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g. SUZLON.NS, IDEA.NS, RELIANCE.NS)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive ticker selection menu")
    args = parser.parse_args()

    if args.ticker:
        target = args.ticker
        print(f"\n[+] Fetching live financial statements for {target} from Yahoo Finance...")
        try:
            res = fetch_and_predict(target)
            display_assessment(res)
        except Exception as e:
            print(f"[!] Error fetching/predicting {target}: {e}")
        return

    # Interactive Mode
    print("\n" + "=" * 65)
    print("    INDIAN CORPORATE DISTRESS PREDICTION - LIVE MARKET TESTER")
    print("=" * 65)
    print("Choose a benchmark Indian stock or enter any custom ticker:")
    for k, (sym, desc) in PRESET_TICKERS.items():
        print(f"  {k}. {sym:<14} : {desc}")
    print("  9. Custom NSE/BSE Ticker (e.g. INFY.NS, TCS.NS, ADANIENT.NS)")
    print("  0. Exit")
    print("-" * 65)

    choice = input("Enter your choice (0-9): ").strip()
    if choice == "0":
        print("Exiting.")
        sys.exit(0)
    elif choice in PRESET_TICKERS:
        target = PRESET_TICKERS[choice][0]
    elif choice == "9":
        target = input("Enter Ticker Symbol (e.g., INFY.NS or INFY): ").strip()
    else:
        print("Invalid option.")
        return

    print(f"\n[+] Connecting to live market data feed for {target}...")
    try:
        res = fetch_and_predict(target)
        display_assessment(res)
    except Exception as e:
        print(f"[!] Failed to audit live ticker {target}: {e}")

if __name__ == "__main__":
    main()
