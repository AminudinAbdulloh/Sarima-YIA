# -*- coding: utf-8 -*-
"""
model_utils.py — SARIMA model fitting, evaluation, and forecasting.

Provides unit root tests (ADF and KPSS), Auto ARIMA estimation,
dynamic generation and evaluation of alternative candidate models,
linear numerical data scaling, and Ljung-Box-based model selection.
"""
import warnings
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss as kpss_test_fn
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from scipy.stats import pearsonr, levene

from config import AUTO_ARIMA_PARAMS, TEST_SIZE, BULAN_FULL

warnings.filterwarnings("ignore")

_SARIMAX_FIT_KWARGS = dict(
    enforce_stationarity=False,
    enforce_invertibility=False,
)


def get_stationarity_info(s: pd.Series) -> dict:
    """Run ADF and KPSS tests on a series."""
    s_clean = s.dropna()
    if len(s_clean) < 10:
        return {
            "adf_stat": 0.0, "adf_pval": 1.0, "adf_ok": False,
            "kpss_stat": 0.0, "kpss_pval": 0.0, "kpss_ok": False
        }
        
    # ADF Test
    adf_res = adfuller(s_clean, autolag="AIC")
    adf_stat = float(adf_res[0])
    adf_pval = float(adf_res[1])
    adf_ok = adf_pval <= 0.05
    
    # KPSS Test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_res = kpss_test_fn(s_clean, regression="c", nlags="auto")
    kpss_stat = float(kpss_res[0])
    kpss_pval = float(kpss_res[1])
    kpss_ok = kpss_pval > 0.05  # KPSS null is stationarity -> p > 0.05 is stationary
    
    return {
        "adf_stat": adf_stat,
        "adf_pval": adf_pval,
        "adf_ok": adf_ok,
        "kpss_stat": kpss_stat,
        "kpss_pval": kpss_pval,
        "kpss_ok": kpss_ok
    }


def map_param_name(name: str) -> str:
    """Map statsmodels parameter names to standard human-readable format (AR1, SMA1, etc.)."""
    name = name.strip()
    if name.startswith("ar.L"):
        return f"AR{name[4:]}"
    if name.startswith("ma.L"):
        return f"MA{name[4:]}"
    if name.startswith("ar.S.L"):
        try:
            lag = int(name[6:])
            return f"SAR{lag // 12}"
        except ValueError:
            return name
    if name.startswith("ma.S.L"):
        try:
            lag = int(name[6:])
            return f"SMA{lag // 12}"
        except ValueError:
            return name
    if name == "intercept":
        return "Intercept"
    if name == "drift":
        return "Drift"
    return name


def run_model(
    ts: pd.Series,
    n_forecast: int,
    use_auto: bool = True,
    manual_order: tuple = (1, 1, 1),
    manual_seasonal: tuple = (1, 1, 1),
) -> dict:
    """
    Runs the entire SARIMA pipeline on raw data using a linear 1:1000 scaling.
    Finds the optimal model using Auto ARIMA, generates 4 neighboring models,
    fits all of them, and selects the winner using Ljung-Box diagnostics.
    """
    # ── Train / test split ─────────────────────────────────────────────────
    train = ts.iloc[:-TEST_SIZE]
    test  = ts.iloc[-TEST_SIZE:]

    # ── Step 1 & 2: Stationarity tests on raw and differenced data ─────────
    stationarity_steps = {
        "Original": get_stationarity_info(train),
        "Diff_NonSeasonal": get_stationarity_info(train.diff().dropna()),
        "Diff_Seasonal": get_stationarity_info(train.diff(12).dropna()),
        "Diff_Both": get_stationarity_info(train.diff().diff(12).dropna()),
    }

    # ── Step 3: Penskalaan linier data (skala 1:1000) ──────────────────────
    train_scaled = train / 1000.0
    test_scaled = test / 1000.0
    ts_scaled = ts / 1000.0

    # Run Auto ARIMA on scaled training data (max_P=1, max_Q=1)
    # Override constraints directly to guarantee bounds
    params = AUTO_ARIMA_PARAMS.copy()
    params["max_P"] = 1
    params["max_Q"] = 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto = pm.auto_arima(train_scaled, **params)
        
    best_order = auto.order
    best_seasonal_order = auto.seasonal_order
    
    p, d, q = best_order
    P, D, Q, s = best_seasonal_order

    # Generate candidate model variations (Winner + 4 neighbors)
    candidates_configs = []
    candidates_configs.append((best_order, best_seasonal_order))

    # Alternate seasonal parameters (max P/Q is 1)
    P_alt = P - 1 if P == 1 else P + 1
    Q_alt = Q - 1 if Q == 1 else Q + 1

    variations = [
        ((p + 1 if p < 3 else p - 1, d, q), (P, D, Q, 12)),
        ((p, d, q + 1 if q < 3 else q - 1), (P, D, Q, 12)),
        ((p, d, q), (P_alt, D, Q, 12)),
        ((p, d, q), (P, D, Q_alt, 12)),
        ((0 if p != 0 else 1, d, 1 if q != 1 else 0), (0, D, 1 if Q != 1 else 0, 12))
    ]

    for ord_val, seas_val in variations:
        ord_val = tuple(max(0, x) for x in ord_val)
        seas_val = (max(0, seas_val[0]), max(0, seas_val[1]), max(0, seas_val[2]), 12)
        if (ord_val, seas_val) not in candidates_configs:
            candidates_configs.append((ord_val, seas_val))

    # Fallbacks respecting bounds (P<=1, Q<=1)
    defaults = [
        ((1, d, 1), (1, D, 1, 12)),
        ((2, d, 1), (0, D, 1, 12)),
        ((0, d, 1), (0, D, 1, 12)),
        ((1, d, 0), (0, D, 1, 12))
    ]
    for ord_val, seas_val in defaults:
        if len(candidates_configs) >= 5:
            break
        if (ord_val, seas_val) not in candidates_configs:
            candidates_configs.append((ord_val, seas_val))

    candidates_configs = candidates_configs[:5]

    # ── Step 4: Fit all candidate models ────────────────────────────────────
    candidates_results = []

    for ord_val, seas_val in candidates_configs:
        model_name = f"SARIMA ({ord_val[0]},{ord_val[1]},{ord_val[2]})({seas_val[0]},{seas_val[1]},{seas_val[2]})"
        
        try:
            # Fit on training data (scaled scale)
            res_train = SARIMAX(
                train_scaled, order=ord_val, seasonal_order=seas_val, **_SARIMAX_FIT_KWARGS
            ).fit(disp=False)
            
            # Forecast on test (scale back)
            fc_test = res_train.get_forecast(steps=TEST_SIZE)
            pred_mean = fc_test.predicted_mean * 1000.0
            
            # Compute evaluation metrics (original scale)
            mae  = mean_absolute_error(test, pred_mean)
            rmse = np.sqrt(mean_squared_error(test, pred_mean))
            mape = np.mean(np.abs((test.values - pred_mean.values) / test.values)) * 100
            acc  = 100.0 - mape
            r, _ = pearsonr(test.values, pred_mean.values)
            
            # Residual diagnostics (training, scale back to original for plot)
            residuals = res_train.resid * 1000.0
            lb = acorr_ljungbox(res_train.resid, lags=[12], return_df=True)
            lb_pval = float(lb["lb_pvalue"].values[0])
            lb_stat = float(lb["lb_stat"].values[0])

            # Fit on full data (scaled scale)
            res_full = SARIMAX(
                ts_scaled, order=ord_val, seasonal_order=seas_val, **_SARIMAX_FIT_KWARGS
            ).fit(disp=False)
            
            fitted_full = res_full.fittedvalues * 1000.0
            fc_future = res_full.get_forecast(steps=n_forecast)
            future_mean = fc_future.predicted_mean * 1000.0
            
            future_idx = pd.date_range(
                ts.index[-1] + pd.DateOffset(months=1), periods=n_forecast, freq="MS"
            )
            future_mean.index = future_idx

            forecast_rows = [
                {
                    "Periode":  d.strftime("%Y-%m"),
                    "Bulan":    f"{BULAN_FULL[d.month]} {d.year}",
                    "Prediksi":  round(float(v)),
                }
                for d, v in zip(future_mean.index, future_mean.values)
            ]

            # Individual coefficients table
            coef_table = []
            for name, coef, stderr, z, pval in zip(
                res_full.param_names,
                res_full.params,
                res_full.bse,
                res_full.tvalues,
                res_full.pvalues
            ):
                mapped_name = map_param_name(name)
                pval_val = float(pval) if not pd.isna(pval) else np.nan
                
                coef_table.append({
                    "Parameter": mapped_name,
                    "Koefisien": float(coef),
                    "Std Error": float(stderr) if not pd.isna(stderr) else np.nan,
                    "z-stat": float(z) if not pd.isna(z) else np.nan,
                    "P>|z|": pval_val,
                    "Signifikansi": "✅ Signifikan (p≤0.05)" if pval_val <= 0.05 else "❌ Tidak Signifikan (p>0.05)"
                })
                
            df_coef = pd.DataFrame(coef_table)

            candidates_results.append({
                "name": model_name,
                "order": ord_val,
                "seasonal_order": seas_val,
                "train": train,
                "test": test,
                "pred_mean": pred_mean,
                "fitted_full": fitted_full,
                "future_mean": future_mean,
                "forecast_rows": forecast_rows,
                "residuals": residuals,
                "mae": mae, "rmse": rmse, "mape": mape, "acc": acc,
                "r": r,
                "aic": float(res_full.aic),
                "bic": float(res_full.bic),
                "lb_stat": lb_stat,
                "lb_p": lb_pval,
                "df_coef": df_coef
            })
        except Exception as e:
            continue

    # ── Step 5: Select Winner based on Ljung-Box & AIC ─────────────────────
    # Choose model passing Ljung-Box (p-value > 0.05) with minimum AIC.
    # Fallback to minimum AIC among all candidates if none pass.
    passing_candidates = [c for c in candidates_results if c["lb_p"] > 0.05]
    if passing_candidates:
        winner_model = min(passing_candidates, key=lambda x: x["aic"])
    else:
        winner_model = min(candidates_results, key=lambda x: x["aic"])

    # Reorder candidates list so that the selected winner model is at index 0
    final_candidates = [winner_model]
    for c in candidates_results:
        if c["name"] != winner_model["name"]:
            final_candidates.append(c)

    # ── Step 6: Construct combined parameter estimates table ───────────────
    combined_rows = []
    for cand in final_candidates:
        # Re-extract coefficients from cand's df_coef (exclude sigma2)
        for _, row in cand["df_coef"].iterrows():
            param_name = row["Parameter"]
            if param_name == "sigma2":
                continue
                
            pval_val = row["P>|z|"]
            
            if pd.isna(pval_val):
                keputusan = "Gagal Tolak H0"
            else:
                keputusan = "Tolak H0" if pval_val <= 0.05 else "Gagal Tolak H0"
                
            combined_rows.append({
                "Model": cand["name"],
                "Koefisien": param_name,
                "P-value": pval_val,
                "Alpha": 0.05,
                "Keputusan": keputusan
            })
            
    df_combined_params = pd.DataFrame(combined_rows)

    return dict(
        stationarity_steps=stationarity_steps,
        candidates=final_candidates,
        df_combined_params=df_combined_params,
        ts_diff_both=train.diff().diff(12).dropna()
    )
