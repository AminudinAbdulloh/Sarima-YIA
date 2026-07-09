# -*- coding: utf-8 -*-
"""
data_utils.py — Data loading, preprocessing, and statistical tests.

All @st.cache_data functions live here so caching is consistent.
"""
import io
import warnings
import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss as kpss_test_fn

warnings.filterwarnings("ignore")


@st.cache_data(show_spinner=False)
def load_and_preprocess(file_bytes: bytes):
    """
    Parse a CSV from raw bytes and compute derived columns.

    Returns
    -------
    df_raw : pd.DataFrame  — full monthly data table
    ts_raw : pd.Series     — raw total passengers time series
    ts     : pd.Series     — working time series (same as ts_raw, kept for API compat)
    """
    df = pd.read_csv(io.BytesIO(file_bytes))

    # Normalisasi nama kolom: lowercase → strip → rename ke PascalCase
    # Agar dataset lama (PascalCase) maupun baru (lowercase) keduanya berjalan
    col_map = {
        "periode":                    "Periode",
        "tahun":                      "Tahun",
        "bulan":                      "Bulan",
        "pesawat_datang":             "Pesawat_Datang",
        "pesawat_berangkat":          "Pesawat_Berangkat",
        "penumpang_datang":           "Penumpang_Datang",
        "penumpang_berangkat":        "Penumpang_Berangkat",
        "penumpang_transit_datang":   "Penumpang_Transit_Datang",
        "penumpang_transit_berangkat":"Penumpang_Transit_Berangkat",
        "penumpang transit_datang":   "Penumpang_Transit_Datang",
        "penumpang transit_berangkat":"Penumpang_Transit_Berangkat",
        "kargo_datang":               "Kargo_Datang",
        "kargo_berangkat":            "Kargo_Berangkat",
        "bagasi_datang":              "Bagasi_Datang",
        "bagasi_berangkat":           "Bagasi_Berangkat",
        "pos_datang":                 "Pos_Datang",
        "pos_berangkat":              "Pos_Berangkat",
    }
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .map(lambda c: col_map.get(c, c))
    )

    df["Periode"] = pd.to_datetime(df["Periode"], format="%Y-%m")
    df = df.sort_values("Periode").reset_index(drop=True)

    # Derived totals
    df["Total_Penumpang"] = df["Penumpang_Datang"] + df["Penumpang_Berangkat"]
    df["Total_Pesawat"]   = df["Pesawat_Datang"]   + df["Pesawat_Berangkat"]
    df["Total_Transit"]   = (
        df["Penumpang_Transit_Datang"] + df["Penumpang_Transit_Berangkat"]
    )

    # Ganti baris dengan Total_Penumpang = 0 atau NaN menjadi NaN
    # (data 2023 banyak yang kosong, perlu interpolasi agar model tidak error)
    df.loc[df["Total_Penumpang"] <= 0, "Total_Penumpang"] = pd.NA
    df["Total_Penumpang"] = (
        df["Total_Penumpang"]
        .astype("float64")
        .interpolate(method="linear", limit_direction="both")
    )

    # Demikian pula untuk kolom turunan lain agar konsisten
    for col in ["Penumpang_Datang", "Penumpang_Berangkat",
                "Total_Pesawat", "Pesawat_Datang", "Pesawat_Berangkat"]:
        df[col] = (
            df[col]
            .astype("float64")
            .replace(0, pd.NA)
            .interpolate(method="linear", limit_direction="both")
        )

    # Drop any partial month beyond the analysis window
    df = df[df["Periode"] < "2026-05-01"].copy()
    df["Tahun"] = df["Periode"].dt.year

    # asfreq("MS") mengisi bulan yang hilang dengan NaN, lalu interpolasi
    ts_raw = (
        df.set_index("Periode")["Total_Penumpang"]
        .asfreq("MS")
        .interpolate(method="linear", limit_direction="both")
    )
    ts = ts_raw.copy()
    return df, ts_raw, ts


@st.cache_data(show_spinner=False)
def run_stationarity(ts_bytes: bytes) -> dict:
    """
    Run ADF and KPSS stationarity tests on three versions of the series:
    original, first-differenced, and seasonally-differenced (lag=12).
    """
    ts = pd.read_json(io.BytesIO(ts_bytes), typ="series")
    ts.index = pd.to_datetime(ts.index)

    series_map = {
        "Original":          ts,
        "Diff-1":            ts.diff().dropna(),
        "Seasonal Diff-12":  ts.diff(12).dropna(),
    }
    results = {}
    for label, s in series_map.items():
        s = s.dropna()
        adf  = adfuller(s, autolag="AIC")
        kpss = kpss_test_fn(s, regression="c", nlags="auto")
        results[label] = {
            "adf_stat":  adf[0],  "adf_pval": adf[1],
            "adf_ok":    adf[1] <= 0.05,
            "adf_crit":  adf[4],
            "kpss_stat": kpss[0], "kpss_pval": kpss[1],
            "kpss_ok":   kpss[1] > 0.05,
        }
    return results


@st.cache_data(show_spinner=False)
def run_decomposition(ts_bytes: bytes) -> dict:
    """Additive seasonal decomposition with period = 12 months."""
    ts = pd.read_json(io.BytesIO(ts_bytes), typ="series")
    ts.index = pd.to_datetime(ts.index)
    decomp = seasonal_decompose(
        ts, model="additive", period=12, extrapolate_trend="freq"
    )
    return {
        "observed": decomp.observed.to_dict(),
        "trend":    decomp.trend.to_dict(),
        "seasonal": decomp.seasonal.to_dict(),
        "resid":    decomp.resid.to_dict(),
    }
