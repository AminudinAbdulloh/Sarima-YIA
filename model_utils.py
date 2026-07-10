# -*- coding: utf-8 -*-
"""
model_utils.py — SARIMA model fitting, evaluation, and forecasting.

Provides unit root tests (ADF and KPSS), Auto ARIMA estimation,
dynamic generation and evaluation of alternative candidate models,
and parameter estimates formatting.
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
        # e.g., ar.S.L12 -> SAR1
        try:
            lag = int(name[6:])
            return f"SAR{lag // 12}"
        except ValueError:
            return name
    if name.startswith("ma.S.L"):
        # e.g., ma.S.L12 -> SMA1, ma.S.L24 -> SMA2
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
    use_auto: bool = True,  # Kept for compatibility, but UI always runs auto
    manual_order: tuple = (1, 1, 1),
    manual_seasonal: tuple = (1, 1, 1),
) -> dict:
    """
    Runs the entire SARIMA pipeline on original scale data.
    Finds the optimal model using Auto ARIMA, generates 4 neighboring models,
    fits all of them, and builds comparative tables.
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

    # ── Step 3: Run Auto ARIMA to find the best model ──────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto = pm.auto_arima(train, **AUTO_ARIMA_PARAMS)
        
    best_order = auto.order
    best_seasonal_order = auto.seasonal_order
    
    p, d, q = best_order
    P, D, Q, s = best_seasonal_order

    # Generate distinct candidates dynamically (Winner + 4 alternative neighbors)
    candidates_configs = []
    candidates_configs.append((best_order, best_seasonal_order))

    variations = [
        ((p + 1 if p < 3 else p - 1, d, q), (P, D, Q, 12)),
        ((p, d, q + 1 if q < 3 else q - 1), (P, D, Q, 12)),
        ((p, d, q), (P + 1 if P < 2 else P - 1, D, Q, 12)),
        ((p, d, q), (P, D, Q + 1 if Q < 2 else Q - 1, 12)),
        ((0 if p != 0 else 1, d, 1 if q != 1 else 0), (0, D, 1 if Q != 1 else 0, 12))
    ]

    for ord_val, seas_val in variations:
        ord_val = tuple(max(0, x) for x in ord_val)
        seas_val = (max(0, seas_val[0]), max(0, seas_val[1]), max(0, seas_val[2]), 12)
        if (ord_val, seas_val) not in candidates_configs:
            candidates_configs.append((ord_val, seas_val))

    # Fallbacks to ensure exactly 5 models if duplicates occur
    defaults = [
        ((1, d, 1), (1, D, 1, 12)),
        ((2, d, 1), (0, D, 1, 12)),
        ((0, d, 1), (0, D, 2, 12)),
        ((1, d, 0), (0, D, 1, 12))
    ]
    for ord_val, seas_val in defaults:
        if len(candidates_configs) >= 5:
            break
        if (ord_val, seas_val) not in candidates_configs:
            candidates_configs.append((ord_val, seas_val))

    candidates_configs = candidates_configs[:5]

    # ── Step 4: Fit all candidate models and compile results ────────────────
    candidates_results = []
    combined_rows = []

    for ord_val, seas_val in candidates_configs:
        model_name = f"SARIMA ({ord_val[0]},{ord_val[1]},{ord_val[2]})({seas_val[0]},{seas_val[1]},{seas_val[2]})"
        
        try:
            # Fit on training data
            res_train = SARIMAX(
                train, order=ord_val, seasonal_order=seas_val, **_SARIMAX_FIT_KWARGS
            ).fit(disp=False)
            
            fc_test = res_train.get_forecast(steps=TEST_SIZE)
            pred_mean = fc_test.predicted_mean
            
            # Compute evaluation metrics
            mae  = mean_absolute_error(test, pred_mean)
            rmse = np.sqrt(mean_squared_error(test, pred_mean))
            mape = np.mean(np.abs((test.values - pred_mean.values) / test.values)) * 100
            acc  = 100.0 - mape
            r, _ = pearsonr(test.values, pred_mean.values)
            
            # Residual diagnostics (training)
            residuals = res_train.resid
            lb = acorr_ljungbox(residuals, lags=[12], return_df=True)
            lb_pval = float(lb["lb_pvalue"].values[0])
            lb_stat = float(lb["lb_stat"].values[0])
            _, norm_p = stats.shapiro(residuals)

            # Fit on full data
            res_full = SARIMAX(
                ts, order=ord_val, seasonal_order=seas_val, **_SARIMAX_FIT_KWARGS
            ).fit(disp=False)
            
            fitted_full = res_full.fittedvalues
            fc_future = res_full.get_forecast(steps=n_forecast)
            future_mean = fc_future.predicted_mean
            
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

            # Build individual coefficients table
            coef_table = []
            for name, coef, stderr, z, pval in zip(
                res_full.param_names,
                res_full.params,
                res_full.bse,
                res_full.tvalues,
                res_full.pvalues
            ):
                is_sigma = (name == "sigma2")
                mapped_name = map_param_name(name)
                
                # Coefficients for individual model table
                coef_table.append({
                    "Parameter": mapped_name,
                    "Koefisien": float(coef),
                    "Std Error": float(stderr),
                    "z-stat": float(z),
                    "P>|z|": float(pval),
                    "Signifikansi": "✅ Signifikan (p≤0.05)" if pval <= 0.05 else "❌ Tidak Signifikan (p>0.05)"
                })
                
                # Rows for combined table (exclude sigma2)
                if not is_sigma:
                    combined_rows.append({
                        "Model": model_name,
                        "Koefisien": mapped_name,
                        "P-value": float(pval),
                        "Alpha": 0.05,
                        "Keputusan": "Tolak H0" if pval <= 0.05 else "Gagal Tolak H0"
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
                "norm_p": float(norm_p),
                "df_coef": df_coef
            })
        except Exception as e:
            # Handle fit failures gracefully
            continue

    # Combined estimates table
    df_combined_params = pd.DataFrame(combined_rows)

    # Return full data structure
    return dict(
        stationarity_steps=stationarity_steps,
        candidates=candidates_results,
        df_combined_params=df_combined_params,
        ts_diff_both=train.diff().diff(12).dropna()
    )
