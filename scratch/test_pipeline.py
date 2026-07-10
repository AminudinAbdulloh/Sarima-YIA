# -*- coding: utf-8 -*-
import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from data_utils import load_and_preprocess
from model_utils import run_model

def test():
    print("Loading data...")
    csv_path = "data_penumpang_domestik_yia.csv"
    with open(csv_path, "rb") as f:
        file_bytes = f.read()
        
    df, ts_raw, ts = load_and_preprocess(file_bytes)
    print(f"Data loaded successfully. Length: {len(ts)}")
    print(f"Index range: {ts.index[0]} to {ts.index[-1]}")
    
    print("\nRunning model pipeline...")
    # Run the new pipeline
    res = run_model(ts, n_forecast=12)
    
    print("\n--- STATIONARITY TESTS ---")
    s_steps = res["stationarity_steps"]
    for label, key in [
        ("Original", "Original"),
        ("Diff Non-Seasonal (d=1, D=0)", "Diff_NonSeasonal"),
        ("Diff Seasonal (d=0, D=1)", "Diff_Seasonal"),
        ("Diff Both (d=1, D=1)", "Diff_Both")
    ]:
        r = s_steps[key]
        print(f"{label}:")
        print(f"  ADF: Stat = {r['adf_stat']:.4f}, p-val = {r['adf_pval']:.4f}, ok = {r['adf_ok']}")
        print(f"  KPSS: Stat = {r['kpss_stat']:.4f}, p-val = {r['kpss_pval']:.4f}, ok = {r['kpss_ok']}")
        
    print("\n--- CANDIDATE MODELS ---")
    candidates = res["candidates"]
    for idx, cand in enumerate(candidates, 1):
        print(f"Model {idx}: {cand['name']}")
        print(f"  AIC = {cand['aic']:.2f}, BIC = {cand['bic']:.2f}, MAPE = {cand['mape']:.2f}%")
        print(f"  Ljung-Box p-val = {cand['lb_p']:.4f}, Shapiro-Wilk p-val = {cand['norm_p']:.4f}")
        
    print("\n--- COMBINED ESTIMATES TABLE (Top 10 rows) ---")
    df_comb = res["df_combined_params"]
    print(df_comb.head(10))
    
    print("\nVerification complete. New multi-model pipeline runs successfully!")

if __name__ == "__main__":
    test()
