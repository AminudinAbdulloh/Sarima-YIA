# -*- coding: utf-8 -*-
"""
model_utils.py — SARIMA model fitting, evaluation, and forecasting.

Parameter constraints are imported from config to stay consistent
with the research's batasan masalah.
"""
import warnings
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from scipy.stats import pearsonr

from config import AUTO_ARIMA_PARAMS, TEST_SIZE, BULAN_FULL

warnings.filterwarnings("ignore")

_SARIMAX_FIT_KWARGS = dict(
    enforce_stationarity=False,
    enforce_invertibility=False,
)


def run_model(
    ts: pd.Series,
    n_forecast: int,
    use_auto: bool,
    manual_order: tuple,
    manual_seasonal: tuple,
) -> dict:
    """
    Fit SARIMA on training data, evaluate on test, then refit on
    full data and produce future forecasts.

    Parameters
    ----------
    ts              : full time series (train + test)
    n_forecast      : months to forecast ahead (6–24)
    use_auto        : if True, run Auto ARIMA; otherwise use manual params
    manual_order    : (p, d, q) when use_auto=False
    manual_seasonal : (P, D, Q) when use_auto=False (s=12 appended automatically)

    Returns
    -------
    dict with all results needed for charts, tables, and diagnostics.
    """
    # ── Train / test split ─────────────────────────────────────────────────
    train = ts.iloc[:-TEST_SIZE]
    test  = ts.iloc[-TEST_SIZE:]

    # ── Parameter selection ────────────────────────────────────────────────
    if use_auto:
        auto           = pm.auto_arima(train, **AUTO_ARIMA_PARAMS)
        order          = auto.order
        seasonal_order = auto.seasonal_order   # already includes m=12
        aic_sel        = auto.aic()
        bic_sel        = auto.bic()
    else:
        order          = manual_order
        seasonal_order = (*manual_seasonal[:3], 12)
        aic_sel = bic_sel = None

    # ── Fit on training data ───────────────────────────────────────────────
    res_train = SARIMAX(
        train, order=order, seasonal_order=seasonal_order, **_SARIMAX_FIT_KWARGS
    ).fit(disp=False)

    fc_test   = res_train.get_forecast(steps=TEST_SIZE)
    pred_mean = fc_test.predicted_mean

    # ── Evaluation metrics ─────────────────────────────────────────────────
    mae  = mean_absolute_error(test, pred_mean)
    rmse = np.sqrt(mean_squared_error(test, pred_mean))
    mape = np.mean(np.abs((test.values - pred_mean.values) / test.values)) * 100
    acc  = 100.0 - mape
    r, _ = pearsonr(test.values, pred_mean.values)
    aic  = res_train.aic if aic_sel is None else aic_sel
    bic  = res_train.bic if bic_sel is None else bic_sel

    # ── Refit on full data → produce future forecast ───────────────────────
    res_full  = SARIMAX(
        ts, order=order, seasonal_order=seasonal_order, **_SARIMAX_FIT_KWARGS
    ).fit(disp=False)

    fc_future   = res_full.get_forecast(steps=n_forecast)
    future_idx  = pd.date_range(
        ts.index[-1] + pd.DateOffset(months=1), periods=n_forecast, freq="MS"
    )
    future_mean = fc_future.predicted_mean
    future_mean.index = future_idx

    forecast_rows = [
        {
            "Periode":    d.strftime("%Y-%m"),
            "Bulan":      f"{BULAN_FULL[d.month]} {d.year}",
            "Prediksi":   round(float(v)),
        }
        for d, v in zip(
            future_mean.index,
            future_mean.values,
        )
    ]

    # ── Residual diagnostics ───────────────────────────────────────────────
    residuals = res_train.resid
    lb        = acorr_ljungbox(residuals, lags=[12], return_df=True)
    _, norm_p = stats.shapiro(residuals)

    return dict(
        order=order,
        seasonal_order=seasonal_order,
        train=train,
        test=test,
        pred_mean=pred_mean,
        fitted_full=res_full.fittedvalues,
        future_mean=future_mean,
        forecast_rows=forecast_rows,
        residuals=residuals,
        mae=mae, rmse=rmse, mape=mape, acc=acc,
        r=r,
        aic=aic, bic=bic,
        lb_stat=float(lb["lb_stat"].values[0]),
        lb_p=float(lb["lb_pvalue"].values[0]),
        norm_p=float(norm_p),
    )
