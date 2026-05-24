# -*- coding: utf-8 -*-
"""
=============================================================================
  FusionSARIMA-UPG  |  Streamlit Dashboard
  Prediksi Jumlah Penumpang Domestik Bandara Sultan Hasanuddin Makassar
=============================================================================
"""

import os
import io
import glob
import warnings
import tempfile
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss as kpss_test_fn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from scipy import stats
from scipy.stats import pearsonr

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FusionSARIMA-UPG | Prediksi Penumpang",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BULAN_FULL = {
    1:"Januari", 2:"Februari", 3:"Maret", 4:"April",
    5:"Mei",     6:"Juni",     7:"Juli",  8:"Agustus",
    9:"September",10:"Oktober",11:"November",12:"Desember"
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0F172A",
    plot_bgcolor="#1A2540",
    font=dict(color="#CBD5E1", family="Inter, sans-serif", size=12),
    margin=dict(l=50, r=30, t=50, b=40),
    legend=dict(
        bgcolor="rgba(30,41,59,0.8)",
        bordercolor="#334155",
        borderwidth=1,
    ),
    xaxis=dict(gridcolor="#1E3050", linecolor="#334155", zeroline=False),
    yaxis=dict(gridcolor="#1E3050", linecolor="#334155", zeroline=False),
)

C = {
    "blue":   "#3B82F6",
    "green":  "#10B981",
    "amber":  "#F59E0B",
    "red":    "#EF4444",
    "purple": "#8B5CF6",
    "cyan":   "#06B6D4",
    "dark":   "#0F172A",
    "card":   "#1E293B",
    "border": "#334155",
    "text":   "#CBD5E1",
    "muted":  "#64748B",
}

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root ── */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#080E1E 0%,#0D1B38 50%,#080E1E 100%) !important;
    font-family:'Inter',sans-serif !important;
}
[data-testid="stMain"] { background: transparent !important; }
[data-testid="block-container"] { padding: 1.5rem 2rem; }

/* ── Hide default elements ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0A1629 0%,#0D1B38 100%) !important;
    border-right: 1px solid #1E3050 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
    color: #64748B;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 16px 0 8px 0;
}
/* ── Radio button nav ── */
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    border-radius:10px !important;
    padding:10px 14px !important;
    transition:all 0.2s ease !important;
    cursor:pointer;
    width:100%;
    color:#94A3B8 !important;
    font-weight:500 !important;
    font-size:14px !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background:rgba(59,130,246,0.12) !important;
    color:#E2E8F0 !important;
}

/* ── Metric Cards ── */
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:24px; }
.kpi-card {
    background:linear-gradient(135deg,#1E293B 0%,#162032 100%);
    border:1px solid #1E3050;
    border-radius:16px;
    padding:20px 24px;
    position:relative;
    overflow:hidden;
    transition:transform 0.25s ease,box-shadow 0.25s ease;
}
.kpi-card:hover { transform:translateY(-4px); box-shadow:0 12px 40px rgba(0,0,0,0.5); }
.kpi-card::before {
    content:'';position:absolute;top:0;left:0;right:0;
    height:3px;border-radius:16px 16px 0 0;
}
.kpi-card.blue::before  { background:linear-gradient(90deg,#2563EB,#60A5FA); }
.kpi-card.green::before { background:linear-gradient(90deg,#059669,#34D399); }
.kpi-card.amber::before { background:linear-gradient(90deg,#D97706,#FCD34D); }
.kpi-card.red::before   { background:linear-gradient(90deg,#DC2626,#F87171); }
.kpi-card.purple::before { background:linear-gradient(90deg,#7C3AED,#C4B5FD); }

.kpi-label {
    font-size:11px;font-weight:700;
    text-transform:uppercase;letter-spacing:0.08em;
    color:#64748B;margin-bottom:8px;
}
.kpi-value { font-size:28px;font-weight:800;color:#F1F5F9;line-height:1.15; }
.kpi-sub { font-size:12px;color:#64748B;margin-top:6px; }
.kpi-icon { position:absolute;top:18px;right:20px;font-size:30px;opacity:0.18; }

/* ── Section Headers ── */
.sec-header {
    display:flex;align-items:center;gap:14px;
    margin:32px 0 20px 0;
    padding-bottom:14px;border-bottom:1px solid #1E3050;
}
.sec-badge {
    background:linear-gradient(135deg,#2563EB,#3B82F6);
    border-radius:12px;width:44px;height:44px;
    display:flex;align-items:center;justify-content:center;
    font-size:20px;flex-shrink:0;
}
.sec-title { font-size:20px;font-weight:700;color:#F1F5F9;margin:0; }
.sec-sub   { font-size:13px;color:#64748B;margin:3px 0 0 0; }

/* ── Info boxes ── */
.box-info {
    background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.25);
    border-radius:12px;padding:14px 18px;margin:10px 0;color:#93C5FD;font-size:13px;
}
.box-success {
    background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);
    border-radius:12px;padding:14px 18px;margin:10px 0;color:#6EE7B7;font-size:13px;
}
.box-warn {
    background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
    border-radius:12px;padding:14px 18px;margin:10px 0;color:#FCD34D;font-size:13px;
}
.box-err {
    background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);
    border-radius:12px;padding:14px 18px;margin:10px 0;color:#FCA5A5;font-size:13px;
}

/* ── Result badges ── */
.badge { display:inline-block;border-radius:20px;padding:3px 13px;font-size:12px;font-weight:700;border:1px solid; }
.badge-ok   { background:rgba(16,185,129,0.15);color:#34D399;border-color:#059669; }
.badge-warn { background:rgba(245,158,11,0.15);color:#FCD34D;border-color:#D97706; }
.badge-err  { background:rgba(239,68,68,0.15);color:#F87171;border-color:#DC2626; }

/* ── Page title ── */
.page-hero {
    background:linear-gradient(135deg,rgba(37,99,235,0.15) 0%,rgba(16,185,129,0.08) 100%);
    border:1px solid rgba(37,99,235,0.2);
    border-radius:20px;padding:28px 36px;margin-bottom:28px;
    position:relative;overflow:hidden;
}
.page-hero::after {
    content:'✈️';position:absolute;right:36px;bottom:10px;
    font-size:80px;opacity:0.07;
}
.page-hero h1 { font-size:26px;font-weight:900;color:#F1F5F9;margin:0 0 6px 0; }
.page-hero p  { font-size:14px;color:#94A3B8;margin:0; }

/* ── Stat table ── */
.stat-table { width:100%;border-collapse:collapse; }
.stat-table th {
    background:#1E293B;color:#64748B;font-size:11px;
    font-weight:700;text-transform:uppercase;letter-spacing:0.06em;
    padding:10px 14px;text-align:left;border-bottom:1px solid #334155;
}
.stat-table td {
    padding:10px 14px;color:#CBD5E1;font-size:13px;
    border-bottom:1px solid rgba(51,65,85,0.4);
}
.stat-table tr:hover td { background:rgba(59,130,246,0.06); }

/* ── Sidebar brand ── */
.sb-brand { text-align:center;padding:20px 12px 16px;border-bottom:1px solid #1E3050;margin-bottom:8px; }
.sb-brand-icon { font-size:36px;margin-bottom:6px; }
.sb-brand h2 {
    font-size:16px;font-weight:800;margin:0;
    background:linear-gradient(90deg,#60A5FA,#34D399);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.sb-brand p { font-size:11px;color:#475569;margin:4px 0 0 0; }

/* ── Separator ── */
.sep { border:none;border-top:1px solid #1E3050;margin:20px 0; }

/* ── Streamlit overrides ── */
.stButton>button {
    border-radius:10px !important;font-weight:600 !important;
    transition:all 0.2s ease !important;
}
.stButton>button:hover { transform:translateY(-2px) !important; }
[data-testid="stDataFrame"] { border-radius:12px !important; overflow:hidden; }
.stTabs [data-baseweb="tab-list"] { background:#1E293B;border-radius:12px;padding:4px; }
.stTabs [data-baseweb="tab"] { border-radius:8px;color:#64748B;font-weight:600; }
.stTabs [aria-selected="true"] { background:#2563EB !important;color:white !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt(n):
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"

def fmt_num(n): return f"{n:,.0f}"
def fmt_pct(n): return f"{n:.2f}%"

def kpi_card(label, value, sub, icon, color="blue"):
    return f"""
<div class="kpi-card {color}">
  <div class="kpi-icon">{icon}</div>
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>"""

def sec_header(icon, title, sub=""):
    st.markdown(f"""
<div class="sec-header">
  <div class="sec-badge">{icon}</div>
  <div>
    <div class="sec-title">{title}</div>
    <div class="sec-sub">{sub}</div>
  </div>
</div>""", unsafe_allow_html=True)

def badge(text, kind="ok"):
    return f'<span class="badge badge-{kind}">{text}</span>'

def apply_plotly_theme(fig, title="", height=420):
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=14, color="#F1F5F9", weight=700), x=0.01),
        height=height,
        hoverlabel=dict(bgcolor="#1E293B", bordercolor="#334155", font_color="#F1F5F9"),
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridwidth=0.5)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(file_bytes: bytes, z_thresh: float = 3.5, iqr_mult: float = 1.5):
    """Load CSV dari bytes, preprocess, detect outliers."""
    df_raw = pd.read_csv(io.BytesIO(file_bytes))
    df_raw["Periode"] = pd.to_datetime(df_raw["Periode"], format="%Y-%m")
    df_raw = df_raw.sort_values("Periode").reset_index(drop=True)
    df_raw["Total_Penumpang"]  = df_raw["Penumpang_Datang"]          + df_raw["Penumpang_Berangkat"]
    df_raw["Total_Pesawat"]    = df_raw["Pesawat_Datang"]            + df_raw["Pesawat_Berangkat"]
    df_raw["Total_Transit"]    = df_raw["Penumpang Transit_Datang"]  + df_raw["Penumpang Transit_Berangkat"]
    # Hapus data parsial Mei 2026
    df_raw = df_raw[df_raw["Periode"] < "2026-05-01"].copy()
    df_raw["Tahun"] = df_raw["Periode"].dt.year
    ts_raw = df_raw.set_index("Periode")["Total_Penumpang"].asfreq("MS")

    # Modified Z-score
    ts_med = ts_raw.median()
    ts_mad = (ts_raw - ts_med).abs().median()
    mod_z  = 0.6745 * (ts_raw - ts_med) / (ts_mad + 1e-9)
    mask_z = mod_z.abs() > z_thresh

    # IQR
    Q1, Q3 = ts_raw.quantile(0.25), ts_raw.quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - iqr_mult * IQR
    upper_fence = Q3 + iqr_mult * IQR
    mask_iqr = (ts_raw < lower_fence) | (ts_raw > upper_fence)

    outlier_mask  = mask_z | mask_iqr
    outlier_dates = ts_raw[outlier_mask].index.tolist()

    # Treatment: interpolasi linear
    ts = ts_raw.copy()
    ts[outlier_mask] = np.nan
    ts = ts.interpolate(method="time").bfill().ffill()

    df_raw["is_outlier"] = df_raw["Periode"].isin(outlier_dates)

    outlier_info = dict(
        dates=outlier_dates, mod_z=mod_z,
        lower_fence=lower_fence, upper_fence=upper_fence,
        Q1=Q1, Q3=Q3, IQR=IQR, ts_med=ts_med, ts_mad=ts_mad,
    )
    return df_raw, ts_raw, ts, outlier_info

@st.cache_data(show_spinner=False)
def run_stationarity(ts_bytes):
    ts = pd.read_json(io.BytesIO(ts_bytes), typ="series")
    ts.index = pd.to_datetime(ts.index)

    results = {}
    for label, series in [
        ("Original", ts),
        ("Diff-1",   ts.diff().dropna()),
        ("Seasonal Diff-12", ts.diff(12).dropna()),
    ]:
        adf_res = adfuller(series.dropna(), autolag="AIC")
        kpss_res = kpss_test_fn(series.dropna(), regression="c", nlags="auto")
        results[label] = {
            "adf_stat":    adf_res[0],
            "adf_pval":    adf_res[1],
            "adf_ok":      adf_res[1] <= 0.05,
            "adf_crit":    adf_res[4],
            "kpss_stat":   kpss_res[0],
            "kpss_pval":   kpss_res[1],
            "kpss_ok":     kpss_res[1] > 0.05,
        }
    return results

@st.cache_data(show_spinner=False)
def run_decomposition(ts_bytes):
    ts = pd.read_json(io.BytesIO(ts_bytes), typ="series")
    ts.index = pd.to_datetime(ts.index)
    decomp = seasonal_decompose(ts, model="additive", period=12, extrapolate_trend="freq")
    return {
        "observed": decomp.observed.to_dict(),
        "trend":    decomp.trend.to_dict(),
        "seasonal": decomp.seasonal.to_dict(),
        "resid":    decomp.resid.to_dict(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# MODEL FUNCTIONS  (stored in session_state, not cached — depends on user params)
# ─────────────────────────────────────────────────────────────────────────────
def run_model(ts, test_size: int, n_forecast: int, use_auto: bool,
              manual_order: tuple, manual_seasonal: tuple):
    """Fit SARIMA dan return dict hasil lengkap."""
    TRAIN_END  = ts.index[-(test_size + 1)]
    TEST_START = ts.index[-test_size]
    train = ts[:TRAIN_END]
    test  = ts[TEST_START:]

    # Parameter selection
    if use_auto:
        auto = pm.auto_arima(
            train,
            start_p=0, max_p=3, start_q=0, max_q=3, d=None,
            start_P=0, max_P=2, start_Q=0, max_Q=2, D=1, m=12,
            seasonal=True, information_criterion="aic",
            stepwise=True, trace=False,
            error_action="ignore", suppress_warnings=True,
        )
        order          = auto.order
        seasonal_order = auto.seasonal_order
        aic_auto = auto.aic()
        bic_auto = auto.bic()
    else:
        order          = manual_order
        seasonal_order = (*manual_seasonal, 12)
        aic_auto = bic_auto = None

    # Fit on train → evaluate on test
    res_train = SARIMAX(
        train, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)

    pred_test = res_train.get_forecast(steps=len(test))
    pred_mean = pred_test.predicted_mean
    pred_ci   = pred_test.conf_int(alpha=0.05)

    mae    = mean_absolute_error(test, pred_mean)
    rmse   = np.sqrt(mean_squared_error(test, pred_mean))
    mape   = np.mean(np.abs((test.values - pred_mean.values) / test.values)) * 100
    acc    = 100 - mape
    r, r_p = pearsonr(test.values, pred_mean.values)

    aic = res_train.aic if aic_auto is None else aic_auto
    bic = res_train.bic if bic_auto is None else bic_auto

    # Refit on full data
    res_full = SARIMAX(
        ts, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)

    future_fc  = res_full.get_forecast(steps=n_forecast)
    future_idx = pd.date_range(ts.index[-1] + pd.DateOffset(months=1), periods=n_forecast, freq="MS")
    future_mean = future_fc.predicted_mean
    future_ci   = future_fc.conf_int(alpha=0.05)
    future_mean.index = future_idx
    future_ci.index   = future_idx

    forecast_rows = []
    for date, val, lo, hi in zip(
        future_mean.index, future_mean.values,
        future_ci.iloc[:, 0].values, future_ci.iloc[:, 1].values,
    ):
        forecast_rows.append({
            "Periode":    date.strftime("%Y-%m"),
            "Bulan":      f"{BULAN_FULL[date.month]} {date.year}",
            "Prediksi":   round(val),
            "Lower_95CI": round(max(lo, 0)),
            "Upper_95CI": round(hi),
        })

    residuals = res_train.resid
    lb = acorr_ljungbox(residuals, lags=[12], return_df=True)
    _, norm_p = stats.shapiro(residuals)

    return dict(
        order=order, seasonal_order=seasonal_order,
        train=train, test=test,
        pred_mean=pred_mean, pred_ci=pred_ci,
        fitted_full=res_full.fittedvalues,
        future_mean=future_mean, future_ci=future_ci,
        forecast_rows=forecast_rows,
        residuals=residuals,
        mae=mae, rmse=rmse, mape=mape, acc=acc, r=r, r_p=r_p,
        aic=aic, bic=bic,
        lb_stat=lb["lb_stat"].values[0], lb_p=lb["lb_pvalue"].values[0],
        norm_p=norm_p,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def chart_ts_overview(df_raw, ts, ts_raw):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_raw.index, y=ts_raw.values, name="Data Asli",
        line=dict(color="#334155", width=1.4, dash="dot"), opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=ts.index, y=ts.values, name="Setelah Treatment",
        line=dict(color=C["blue"], width=2.2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
    ))
    fig.add_vrect(
        x0="2020-03-01", x1="2020-12-31",
        fillcolor="rgba(239,68,68,0.06)", line_width=0,
        annotation_text="COVID-19", annotation_position="top left",
        annotation_font=dict(color="#F87171", size=11),
    )
    fig.add_vline(x="2022-01-01", line=dict(color=C["green"], width=1.2, dash="dash"))
    apply_plotly_theme(fig, "📈 Total Penumpang Domestik per Bulan", height=400)
    fig.update_yaxes(tickformat=",.0f")
    return fig

def chart_datang_berangkat(df_raw):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_raw["Periode"], y=df_raw["Penumpang_Datang"],
        name="Penumpang Datang", line=dict(color=C["blue"], width=2.2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=df_raw["Periode"], y=df_raw["Penumpang_Berangkat"],
        name="Penumpang Berangkat", line=dict(color=C["amber"], width=2.2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
    ))
    apply_plotly_theme(fig, "🛬🛫 Penumpang Datang vs Berangkat", height=360)
    fig.update_yaxes(tickformat=",.0f")
    return fig

def chart_pesawat(df_raw):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_raw["Periode"], y=df_raw["Total_Pesawat"],
        name="Pergerakan Pesawat",
        marker=dict(color=C["green"], opacity=0.8, line=dict(width=0)),
    ))
    apply_plotly_theme(fig, "✈️ Total Pergerakan Pesawat per Bulan", height=300)
    return fig

def chart_outlier(ts_raw, ts, outlier_info):
    od = outlier_info
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Data Asli + Batas Outlier", "Sebelum vs Sesudah Treatment"])
    # Panel 1
    fig.add_trace(go.Scatter(x=ts_raw.index, y=ts_raw.values, name="Data Asli",
        line=dict(color=C["blue"], width=2)), row=1, col=1)
    fig.add_hline(y=od["upper_fence"], line=dict(color=C["red"], dash="dash", width=1.5), row=1, col=1)
    fig.add_hline(y=od["lower_fence"], line=dict(color=C["red"], dash="dot",  width=1.5), row=1, col=1)
    if od["dates"]:
        fig.add_trace(go.Scatter(
            x=od["dates"], y=ts_raw[od["dates"]].values,
            name="Outlier", mode="markers",
            marker=dict(color=C["red"], size=10, symbol="x"),
        ), row=1, col=1)
    # Panel 2
    fig.add_trace(go.Scatter(x=ts_raw.index, y=ts_raw.values, name="Sebelum",
        line=dict(color="#64748B", dash="dot", width=1.5), opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Sesudah Interpolasi",
        line=dict(color=C["green"], width=2.2)), row=2, col=1)
    if od["dates"]:
        fig.add_trace(go.Scatter(
            x=od["dates"], y=ts[od["dates"]].values,
            name="Nilai Interpolasi", mode="markers",
            marker=dict(color=C["green"], size=10, symbol="triangle-up"),
        ), row=2, col=1)
    apply_plotly_theme(fig, "", height=600)
    fig.update_layout(title=dict(text="🔍 Deteksi & Treatment Outlier (Modified Z-score + IQR)"))
    return fig

def chart_decomp(decomp_dict):
    obs  = pd.Series(decomp_dict["observed"]);  obs.index  = pd.to_datetime(obs.index)
    trnd = pd.Series(decomp_dict["trend"]);     trnd.index = pd.to_datetime(trnd.index)
    seas = pd.Series(decomp_dict["seasonal"]);  seas.index = pd.to_datetime(seas.index)
    res  = pd.Series(decomp_dict["resid"]);     res.index  = pd.to_datetime(res.index)

    fig = make_subplots(rows=4, cols=1, subplot_titles=[
        "Data Asli", "Komponen Trend", "Komponen Musiman (Seasonal)", "Komponen Residual"],
        shared_xaxes=True)
    colors = [C["blue"], C["amber"], C["green"], C["red"]]
    for i, (s, c) in enumerate(zip([obs, trnd, seas, res], colors), 1):
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=fig.layout.annotations[i-1].text,
            line=dict(color=c, width=1.8), fill="tozeroy" if i==1 else "none",
            fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:],16)},0.07)" if i==1 else None,
        ), row=i, col=1)
    if obs is not None:
        fig.add_hline(y=0, line=dict(color="#334155", dash="dash", width=0.8), row=4, col=1)
    apply_plotly_theme(fig, "📉 Dekomposisi Time Series (Additive, s=12)", height=700)
    return fig

def chart_acf_pacf_matplotlib(ts):
    """ACF & PACF menggunakan matplotlib (lebih akurat untuk confidence band)."""
    ts_diff = ts.diff().diff(12).dropna()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.set_facecolor("#0F172A")
    for ax in axes.flat:
        ax.set_facecolor("#1A2540")
        ax.tick_params(colors="#94A3B8"); ax.xaxis.label.set_color("#94A3B8")
        ax.yaxis.label.set_color("#94A3B8"); ax.title.set_color("#F1F5F9")
        for sp in ax.spines.values(): sp.set_color("#1E3050")
        ax.grid(color="#1E3050", linewidth=0.5)

    kwargs = dict(alpha=0.05, lags=30)
    plot_acf(ts,       ax=axes[0, 0], **kwargs, color=C["blue"])
    axes[0, 0].set_title("ACF — Data Asli", color="#F1F5F9", fontweight="bold")
    plot_pacf(ts,      ax=axes[0, 1], **kwargs, color=C["blue"])
    axes[0, 1].set_title("PACF — Data Asli", color="#F1F5F9", fontweight="bold")
    plot_acf(ts_diff,  ax=axes[1, 0], **kwargs, color=C["green"])
    axes[1, 0].set_title("ACF — Setelah Diff (d=1, D=1)", color="#F1F5F9", fontweight="bold")
    plot_pacf(ts_diff, ax=axes[1, 1], **kwargs, color=C["green"])
    axes[1, 1].set_title("PACF — Setelah Diff (d=1, D=1)", color="#F1F5F9", fontweight="bold")

    for ax in axes.flat: ax.set_xlabel("Lag", color="#94A3B8")
    plt.tight_layout()
    return fig

def chart_forecast(ts, m):
    fig = go.Figure()
    train, test = m["train"], m["test"]
    pred_mean, pred_ci   = m["pred_mean"], m["pred_ci"]
    future_mean, future_ci = m["future_mean"], m["future_ci"]
    fitted = m["fitted_full"]

    # Training area
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Training Data",
        line=dict(color=C["blue"], width=2), fill="tozeroy",
        fillcolor="rgba(59,130,246,0.05)"))
    # Fitted
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values, name="Fitted SARIMA",
        line=dict(color=C["cyan"], width=1.2, dash="dot"), opacity=0.7))
    # Test actual
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Aktual (Test)",
        line=dict(color="#CBD5E1", width=2.2), mode="lines+markers",
        marker=dict(size=6, color="#CBD5E1")))
    # Test prediction + CI
    fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 1], mode="lines",
        line=dict(width=0), showlegend=False, name="CI Atas"))
    fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:, 0], mode="lines",
        fill="tonexty", fillcolor="rgba(245,158,11,0.12)",
        line=dict(width=0), showlegend=True, name="95% CI (Test)"))
    fig.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean.values,
        name=f"Prediksi Test (MAPE={m['mape']:.2f}%)",
        line=dict(color=C["amber"], width=2.2), mode="lines+markers",
        marker=dict(size=6, color=C["amber"])))
    # Future prediction + CI
    fig.add_trace(go.Scatter(x=future_ci.index, y=future_ci.iloc[:, 1], mode="lines",
        line=dict(width=0), showlegend=False, name="CI Atas Forecast"))
    fig.add_trace(go.Scatter(x=future_ci.index, y=future_ci.iloc[:, 0], mode="lines",
        fill="tonexty", fillcolor="rgba(16,185,129,0.12)",
        line=dict(width=0), showlegend=True, name="95% CI (Forecast)"))
    fig.add_trace(go.Scatter(x=future_mean.index, y=future_mean.values,
        name="Prediksi ke Depan",
        line=dict(color=C["green"], width=2.5),
        mode="lines+markers+text",
        marker=dict(size=7, color=C["green"]),
        text=[f"{v/1000:.0f}K" for v in future_mean.values],
        textposition="top center", textfont=dict(size=10, color=C["green"]),
    ))
    # Vertical separators
    fig.add_vline(x=test.index[0], line=dict(color=C["amber"], width=1, dash="dot"))
    fig.add_vline(x=future_mean.index[0], line=dict(color=C["green"], width=1.5, dash="dashdot"))

    apply_plotly_theme(fig, "📊 Historis, Evaluasi & Prediksi SARIMA", height=520)
    fig.update_yaxes(tickformat=",.0f")
    return fig

def chart_actual_vs_pred(test, pred_mean, mape, r):
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Line: Aktual vs Prediksi", "Scatter: Aktual vs Prediksi"])
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Aktual",
        line=dict(color="#CBD5E1", width=2), mode="lines+markers",
        marker=dict(size=7)), row=1, col=1)
    fig.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean.values, name="Prediksi",
        line=dict(color=C["amber"], width=2, dash="dash"), mode="lines+markers",
        marker=dict(size=7, symbol="square")), row=1, col=1)
    # Scatter
    lims = [min(test.min(), pred_mean.min())*0.92, max(test.max(), pred_mean.max())*1.07]
    fig.add_trace(go.Scatter(x=test.values, y=pred_mean.values, name="Pasangan Nilai",
        mode="markers", marker=dict(color=C["blue"], size=9, opacity=0.85)), row=1, col=2)
    fig.add_trace(go.Scatter(x=lims, y=lims, name="Garis Sempurna (45°)",
        line=dict(color=C["red"], dash="dash", width=1.5)), row=1, col=2)
    apply_plotly_theme(fig, "✅ Validasi: Aktual vs Prediksi (Periode Test)", height=420)
    fig.update_xaxes(tickformat=",.0f", row=1, col=2)
    fig.update_yaxes(tickformat=",.0f", row=1, col=2)
    return fig

def chart_seasonal_pattern(ts):
    monthly_avg = ts.groupby(ts.index.month).mean()
    avg_all     = monthly_avg.mean()
    colors      = [C["green"] if v >= avg_all else C["amber"] for v in monthly_avg.values]
    bulan_names = [BULAN_FULL[m][:3] for m in monthly_avg.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=bulan_names, y=monthly_avg.values,
        marker=dict(color=colors, line=dict(width=0)),
        name="Rata-rata Penumpang",
        text=[f"{v/1000:.0f}K" for v in monthly_avg.values],
        textposition="outside", textfont=dict(size=11, color="#CBD5E1"),
    ))
    fig.add_hline(y=avg_all, line=dict(color=C["red"], dash="dash", width=2),
        annotation_text=f"Rata-rata: {fmt(avg_all)}", annotation_font=dict(color=C["red"]))
    apply_plotly_theme(fig, "📅 Pola Musiman Bulanan — Rata-rata Total Penumpang", height=380)
    return fig

def chart_residual_diagnostics(residuals):
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Residual vs Waktu", "Histogram Residual", "Q-Q Plot", "ACF Residual"])

    # Residual vs Time
    fig.add_trace(go.Scatter(x=residuals.index, y=residuals.values, name="Residual",
        line=dict(color=C["red"], width=1.5)), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#64748B", dash="dash"), row=1, col=1)

    # Histogram
    fig.add_trace(go.Histogram(x=residuals.values, nbinsx=15,
        marker=dict(color=C["blue"], opacity=0.8, line=dict(color="#0F172A", width=0.8)),
        name="Histogram"), row=1, col=2)
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
    scale  = len(residuals) * (residuals.max() - residuals.min()) / 15
    fig.add_trace(go.Scatter(x=x_norm, y=y_norm*scale, name="Distribusi Normal",
        line=dict(color=C["red"], width=2)), row=1, col=2)

    # Q-Q
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
        marker=dict(color=C["blue"], size=6), name="Q-Q Data"), row=2, col=1)
    x_line = np.array([min(osm), max(osm)])
    fig.add_trace(go.Scatter(x=x_line, y=slope*x_line+intercept,
        line=dict(color=C["amber"], width=2), name="Garis Normal"), row=2, col=1)

    # ACF (matplotlib → embed as image)
    # We'll use a bar chart approximation for ACF
    from statsmodels.tsa.stattools import acf as acf_fn
    acf_vals = acf_fn(residuals, nlags=24, fft=True)
    n = len(residuals)
    conf = 1.96 / np.sqrt(n)
    lag_x = list(range(len(acf_vals)))
    bar_colors = [C["green"] if abs(v) <= conf else C["red"] for v in acf_vals]
    fig.add_trace(go.Bar(x=lag_x, y=acf_vals,
        marker=dict(color=bar_colors, line=dict(width=0)),
        name="ACF Residual"), row=2, col=2)
    fig.add_hline(y=conf,  line=dict(color="#64748B", dash="dot"), row=2, col=2)
    fig.add_hline(y=-conf, line=dict(color="#64748B", dash="dot"), row=2, col=2)
    fig.add_hline(y=0,     line=dict(color="#334155"), row=2, col=2)

    apply_plotly_theme(fig, "🧪 Diagnostik Residual Model SARIMA", height=680)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PAGE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard():
    m = st.session_state.get("model_result")
    df_raw, ts_raw, ts, outlier_info = st.session_state["data"]

    st.markdown("""
<div class="page-hero">
  <h1>✈️ FusionSARIMA-UPG Dashboard</h1>
  <p>Prediksi Penumpang Domestik · Bandara Sultan Hasanuddin Makassar · Metode SARIMA</p>
</div>""", unsafe_allow_html=True)

    # KPI cards
    total_hist = ts.sum()
    n_outlier  = len(outlier_info["dates"])
    n_obs      = len(ts)

    if m:
        cards_html = f"""
<div class="kpi-grid">
  {kpi_card("Total Data Historis", fmt(total_hist), f"{n_obs} bulan observasi", "📦", "blue")}
  {kpi_card("Prediksi 12 Bln ke Depan", fmt(sum(r['Prediksi'] for r in m['forecast_rows'])), "Total estimasi penumpang", "🔮", "green")}
  {kpi_card("Akurasi Model", fmt_pct(m['acc']), f"MAPE: {m['mape']:.2f}%", "🎯", "amber")}
  {kpi_card("Pearson R²", f"{m['r']**2:.4f}", f"R = {m['r']:.4f}", "📐", "purple")}
</div>"""
    else:
        cards_html = f"""
<div class="kpi-grid">
  {kpi_card("Total Data Historis", fmt(total_hist), f"{n_obs} bulan observasi", "📦", "blue")}
  {kpi_card("Outlier Terdeteksi", str(n_outlier), "Telah diinterpolasi", "🔍", "red")}
  {kpi_card("Periode Data", ts.index[0].strftime("%b %Y"), f"s/d {ts.index[-1].strftime('%b %Y')}", "📅", "amber")}
  {kpi_card("Status Model", "Belum Dijalankan", "Buka halaman Model & Prediksi", "⚠️", "amber")}
</div>"""
    st.markdown(cards_html, unsafe_allow_html=True)

    # Main chart
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(chart_ts_overview(df_raw, ts, ts_raw), use_container_width=True)
    with col2:
        sec_header("📅", "Pola Musiman", "Rata-rata per bulan")
        st.plotly_chart(chart_seasonal_pattern(ts), use_container_width=True)

    # Yearly summary
    sec_header("📋", "Ringkasan Tahunan", "Total penumpang per tahun")
    yearly = df_raw.groupby("Tahun")["Total_Penumpang"].agg(["sum", "mean", "min", "max"])
    yearly.columns = ["Total", "Rata-rata/Bln", "Minimum", "Maksimum"]
    yearly_fmt = yearly.applymap(lambda x: fmt_num(round(x))).reset_index()
    st.dataframe(yearly_fmt, use_container_width=True, hide_index=True)

    if m:
        sec_header("🔮", "Ringkasan Prediksi 12 Bulan", f"Model: SARIMA{m['order']}×{m['seasonal_order'][:3]}[s=12]")
        df_fc = pd.DataFrame(m["forecast_rows"])
        df_fc_disp = df_fc.copy()
        for col in ["Prediksi", "Lower_95CI", "Upper_95CI"]:
            df_fc_disp[col] = df_fc_disp[col].apply(fmt_num)
        st.dataframe(df_fc_disp[["Bulan", "Prediksi", "Lower_95CI", "Upper_95CI"]],
                     use_container_width=True, hide_index=True)


def page_eda():
    df_raw, ts_raw, ts, _ = st.session_state["data"]
    sec_header("📊", "Eksplorasi Data (EDA)", "Analisis deskriptif data penumpang Bandara UPG")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Periode Awal",  ts.index[0].strftime("%B %Y"))
    c2.metric("Periode Akhir", ts.index[-1].strftime("%B %Y"))
    c3.metric("Total Observasi", f"{len(ts)} bulan")
    c4.metric("Rata-rata/Bln",   fmt(ts.mean()))

    tabs = st.tabs(["📈 Overview", "🛬🛫 Datang & Berangkat", "✈️ Pergerakan Pesawat", "📋 Data Mentah"])
    with tabs[0]:
        st.plotly_chart(chart_ts_overview(df_raw, ts, ts_raw), use_container_width=True)
    with tabs[1]:
        st.plotly_chart(chart_datang_berangkat(df_raw), use_container_width=True)
    with tabs[2]:
        st.plotly_chart(chart_pesawat(df_raw), use_container_width=True)
    with tabs[3]:
        tahun_list = sorted(df_raw["Tahun"].unique(), reverse=True)
        sel_tahun  = st.multiselect("Filter Tahun", tahun_list, default=tahun_list, key="eda_filter")
        disp_df = df_raw[df_raw["Tahun"].isin(sel_tahun)][
            ["Periode", "Total_Penumpang", "Penumpang_Datang", "Penumpang_Berangkat",
             "Total_Pesawat", "Total_Transit"]].copy()
        disp_df["Periode"] = disp_df["Periode"].dt.strftime("%Y-%m")
        st.dataframe(disp_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    sec_header("📋", "Statistik Tahunan", "Agregasi total penumpang per tahun")
    yearly = df_raw.groupby("Tahun")["Total_Penumpang"].agg(
        Total="sum", Rata2_Bulan="mean", Min="min", Max="max").reset_index()
    yearly_fmt = yearly.copy()
    for c in ["Total", "Rata2_Bulan", "Min", "Max"]:
        yearly_fmt[c] = yearly_fmt[c].apply(lambda x: fmt_num(round(x)))
    st.dataframe(yearly_fmt, use_container_width=True, hide_index=True)


def page_outlier():
    _, ts_raw, ts, outlier_info = st.session_state["data"]
    file_bytes = st.session_state["file_bytes"]

    sec_header("🔍", "Deteksi & Treatment Outlier", "Modified Z-score + IQR")

    with st.expander("⚙️ Konfigurasi Threshold Outlier", expanded=False):
        c1, c2 = st.columns(2)
        z_thresh   = c1.slider("Modified Z-score Threshold", 2.0, 5.0, 3.5, 0.1)
        iqr_mult   = c2.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
        if st.button("🔄 Terapkan Ulang", type="primary"):
            st.session_state["outlier_config"] = (z_thresh, iqr_mult)
            load_and_preprocess.clear()

    z_thresh, iqr_mult = st.session_state.get("outlier_config", (3.5, 1.5))
    _, ts_raw2, ts2, oi = load_and_preprocess(file_bytes, z_thresh, iqr_mult)

    od = oi
    c1, c2, c3 = st.columns(3)
    c1.metric("Outlier Terdeteksi",   f"{len(od['dates'])} bulan")
    c2.metric("Batas Bawah IQR",      fmt_num(od["lower_fence"]))
    c3.metric("Batas Atas IQR",       fmt_num(od["upper_fence"]))

    st.plotly_chart(chart_outlier(ts_raw2, ts2, od), use_container_width=True)

    if od["dates"]:
        sec_header("📋", "Detail Outlier Terdeteksi")
        rows = []
        for d in od["dates"]:
            rows.append({
                "Periode":       d.strftime("%Y-%m"),
                "Bulan":         d.strftime("%B %Y"),
                "Nilai Asli":    fmt_num(int(ts_raw2[d])),
                "Nilai Interpolasi": fmt_num(int(ts2[d])),
                "Mod Z-score":   f"{od['mod_z'][d]:.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(f"""
<div class="box-info">
  💡 <b>Metode:</b> Outlier dideteksi menggunakan gabungan <b>Modified Z-score (Iglewicz &amp; Hoaglin, 1993)</b>
  dengan threshold {z_thresh} dan <b>IQR ×{iqr_mult}</b>.
  Nilai outlier diganti dengan <b>interpolasi linear berbasis waktu (time-based)</b>.
</div>""", unsafe_allow_html=True)


def page_stationarity():
    _, _, ts, _ = st.session_state["data"]
    file_bytes   = st.session_state["file_bytes"]

    sec_header("📈", "Uji Stasioneritas", "ADF Test + KPSS Test + Dekomposisi Time Series")

    # Stationarity tests
    ts_bytes = ts.to_json().encode()
    with st.spinner("Menghitung uji stasioneritas..."):
        stat_res = run_stationarity(ts_bytes)

    tabs = st.tabs(["🧮 ADF + KPSS Tests", "📉 Dekomposisi", "📊 ACF & PACF"])

    with tabs[0]:
        st.markdown("### Hasil Uji Stasioneritas")
        rows = []
        for label, r in stat_res.items():
            rows.append({
                "Series":       label,
                "ADF Stat":     f"{r['adf_stat']:.4f}",
                "ADF p-value":  f"{r['adf_pval']:.4f}",
                "ADF Status":   "✅ Stasioner" if r["adf_ok"] else "❌ Tidak Stasioner",
                "KPSS Stat":    f"{r['kpss_stat']:.4f}",
                "KPSS p-value": f"{r['kpss_pval']:.4f}",
                "KPSS Status":  "✅ Stasioner" if r["kpss_ok"] else "❌ Tidak Stasioner",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        for label, r in stat_res.items():
            with st.expander(f"📋 Detail: {label}", expanded=label=="Original"):
                c1, c2 = st.columns(2)
                c1.markdown(f"""**ADF Test**
- Statistik: `{r['adf_stat']:.4f}`
- p-value: `{r['adf_pval']:.4f}`
- **{"✅ STASIONER (p ≤ 0.05)" if r['adf_ok'] else "❌ TIDAK STASIONER (p > 0.05)"}**""")
                c2.markdown(f"""**KPSS Test**
- Statistik: `{r['kpss_stat']:.4f}`
- p-value: `{r['kpss_pval']:.4f}`
- **{"✅ STASIONER (p > 0.05)" if r['kpss_ok'] else "❌ TIDAK STASIONER (p ≤ 0.05)"}**""")

        st.markdown("""
<div class="box-info">
💡 <b>Interpretasi:</b> ADF Test — H₀: data <i>tidak</i> stasioner (tolak H₀ jika p ≤ 0.05).
KPSS Test — H₀: data <i>stasioner</i> (tolak H₀ jika p ≤ 0.05).
Keduanya harus sepakat untuk konfirmasi stasioneritas.
</div>""", unsafe_allow_html=True)

    with tabs[1]:
        with st.spinner("Menghitung dekomposisi..."):
            decomp_dict = run_decomposition(ts_bytes)
        st.plotly_chart(chart_decomp(decomp_dict), use_container_width=True)
        st.markdown("""
<div class="box-info">
💡 <b>Dekomposisi Aditif:</b> Nilai = Trend + Seasonal + Residual.
Metode ini sesuai ketika amplitudo musiman relatif konstan sepanjang waktu.
</div>""", unsafe_allow_html=True)

    with tabs[2]:
        with st.spinner("Membuat plot ACF & PACF..."):
            fig_acf = chart_acf_pacf_matplotlib(ts)
        st.pyplot(fig_acf, use_container_width=True)
        plt.close("all")
        st.markdown("""
<div class="box-info">
💡 <b>Panduan:</b> ACF membantu menentukan orde MA (q & Q).
PACF membantu menentukan orde AR (p & P).
Garis putus-putus = batas kepercayaan 95%.
</div>""", unsafe_allow_html=True)


def page_model():
    _, _, ts, _ = st.session_state["data"]

    sec_header("🤖", "Model SARIMA & Prediksi", "Auto ARIMA + Fitting + Forecast")

    # Config panel
    with st.expander("⚙️ Konfigurasi Model", expanded=True):
        c1, c2, c3 = st.columns(3)
        use_auto   = c1.toggle("🪄 Auto ARIMA", value=True)
        test_size  = c2.radio("Periode Test", [12, 6], captions=["12 bulan", "6 bulan"], index=0)
        n_forecast = c3.slider("Jumlah Bulan Prediksi", 6, 24, 12)

        if not use_auto:
            st.markdown("**Parameter Manual SARIMA**")
            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            p = mc1.number_input("p", 0, 4, 1)
            d = mc2.number_input("d", 0, 2, 1)
            q = mc3.number_input("q", 0, 4, 1)
            P = mc4.number_input("P", 0, 2, 1)
            D = mc5.number_input("D", 0, 1, 1)
            Q = mc6.number_input("Q", 0, 2, 1)
            manual_order    = (int(p), int(d), int(q))
            manual_seasonal = (int(P), int(D), int(Q))
        else:
            manual_order    = (1, 1, 1)
            manual_seasonal = (1, 1, 1)

    col_run, col_clear = st.columns([1, 4])
    with col_run:
        run_btn = st.button("🚀 Jalankan Model", type="primary", use_container_width=True)
    with col_clear:
        if st.session_state.get("model_result"):
            if st.button("🗑️ Reset Hasil", use_container_width=False):
                st.session_state["model_result"] = None
                st.rerun()

    if run_btn:
        with st.spinner("⏳ Menjalankan model... Auto ARIMA sedang mencari parameter terbaik, harap tunggu."):
            try:
                result = run_model(ts, test_size, n_forecast, use_auto, manual_order, manual_seasonal)
                st.session_state["model_result"] = result
                st.success("✅ Model berhasil dijalankan!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return

    m = st.session_state.get("model_result")
    if not m:
        st.markdown("""
<div class="box-warn">
⚠️ Model belum dijalankan. Klik tombol <b>🚀 Jalankan Model</b> di atas untuk mulai.
<br>Proses Auto ARIMA mungkin memakan waktu 1–5 menit tergantung data.
</div>""", unsafe_allow_html=True)
        return

    # Results
    st.markdown("---")
    sec_header("📌", "Parameter Model Terpilih")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Order (p,d,q)",       str(m["order"]))
    c2.metric("Seasonal (P,D,Q,s)",  str(m["seasonal_order"]))
    c3.metric("AIC",                 f"{m['aic']:.2f}")
    c4.metric("BIC",                 f"{m['bic']:.2f}")

    sec_header("📊", "Hasil Evaluasi Model")
    cards_html = f"""
<div class="kpi-grid">
  {kpi_card("MAE", fmt_num(round(m['mae'])), "Mean Absolute Error", "📏", "blue")}
  {kpi_card("RMSE", fmt_num(round(m['rmse'])), "Root Mean Squared Error", "📐", "purple")}
  {kpi_card("MAPE", fmt_pct(m['mape']), "Mean Absolute % Error", "📉", "amber")}
  {kpi_card("Akurasi", fmt_pct(m['acc']), f"Pearson R = {m['r']:.4f}", "🎯", "green")}
</div>"""
    st.markdown(cards_html, unsafe_allow_html=True)

    # Main forecast chart
    st.plotly_chart(chart_forecast(ts, m), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        # Actual vs predicted chart
        st.plotly_chart(chart_actual_vs_pred(m["test"], m["pred_mean"], m["mape"], m["r"]),
                        use_container_width=True)
    with col_b:
        sec_header("🔮", "Tabel Prediksi ke Depan")
        df_fc = pd.DataFrame(m["forecast_rows"])
        df_fc_disp = df_fc.copy()
        for col in ["Prediksi", "Lower_95CI", "Upper_95CI"]:
            df_fc_disp[col] = df_fc_disp[col].apply(fmt_num)
        st.dataframe(df_fc_disp[["Bulan", "Prediksi", "Lower_95CI", "Upper_95CI"]],
                     use_container_width=True, hide_index=True)
        total_pred = sum(r["Prediksi"] for r in m["forecast_rows"])
        st.markdown(f"""
<div class="box-success">
  ✅ <b>Total Prediksi {n_forecast} Bulan:</b> {fmt_num(total_pred)} penumpang
</div>""", unsafe_allow_html=True)


def page_diagnostics():
    m = st.session_state.get("model_result")
    sec_header("🧪", "Diagnostik Residual", "Validasi asumsi model SARIMA")

    if not m:
        st.markdown("""
<div class="box-warn">
⚠️ Model belum dijalankan. Silakan buka halaman <b>🤖 Model &amp; Prediksi</b> terlebih dahulu.
</div>""", unsafe_allow_html=True)
        return

    residuals = m["residuals"]

    # Summary test results
    lb_ok   = m["lb_p"]   > 0.05
    norm_ok = m["norm_p"] > 0.05

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rata-rata Residual", f"{residuals.mean():.2f}", help="Idealnya mendekati 0")
    c2.metric("Std Residual",       f"{residuals.std():.2f}")
    c3.metric("Ljung-Box p-value",  f"{m['lb_p']:.4f}", delta="OK" if lb_ok else "Perlu Cek",
              delta_color="normal" if lb_ok else "inverse")
    c4.metric("Shapiro-Wilk p-value", f"{m['norm_p']:.4f}", delta="Hampir Normal" if norm_ok else "Tidak Normal",
              delta_color="normal")

    st.plotly_chart(chart_residual_diagnostics(residuals), use_container_width=True)

    st.markdown("### 📋 Interpretasi Diagnostik")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"""**Ljung-Box Test (Autokorelasi Residual)**
- Statistik: `{m['lb_stat']:.4f}`
- p-value: `{m['lb_p']:.4f}`
- Kesimpulan: **{"✅ Tidak ada autokorelasi signifikan (OK)" if lb_ok else "⚠️ Masih ada autokorelasi"}**

> H₀: Tidak ada autokorelasi di residual → Tolak H₀ jika p ≤ 0.05""")
    with cols[1]:
        st.markdown(f"""**Shapiro-Wilk Test (Normalitas Residual)**
- p-value: `{m['norm_p']:.4f}`
- Kesimpulan: **{"✅ Residual berdistribusi Normal" if norm_ok else "ℹ️ Tidak normal (umum di time series)"}**

> H₀: Residual berdistribusi normal → Tolak H₀ jika p ≤ 0.05""")

    if lb_ok:
        st.markdown('<div class="box-success">✅ Residual model <b>white noise</b> — tidak ada pola tersisa yang belum ditangkap model.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="box-warn">⚠️ Masih ada autokorelasi di residual. Pertimbangkan menambah orde AR/MA.</div>',
                    unsafe_allow_html=True)


def page_download():
    m   = st.session_state.get("model_result")
    res = st.session_state.get("data")
    sec_header("📥", "Download Hasil", "Export prediksi, evaluasi, dan data historis")

    if not m:
        st.markdown("""
<div class="box-warn">
⚠️ Model belum dijalankan. Silakan buka halaman <b>🤖 Model &amp; Prediksi</b> terlebih dahulu
agar hasil tersedia untuk diunduh.
</div>""", unsafe_allow_html=True)
        return

    df_raw, ts_raw, ts, outlier_info = res

    # Prepare dataframes
    df_forecast = pd.DataFrame(m["forecast_rows"])
    df_eval = pd.DataFrame({
        "Periode":     m["test"].index.strftime("%Y-%m"),
        "Aktual":      m["test"].values.astype(int),
        "Prediksi":    m["pred_mean"].values.astype(int),
        "CI_Bawah_95": m["pred_ci"].iloc[:, 0].values.astype(int),
        "CI_Atas_95":  m["pred_ci"].iloc[:, 1].values.astype(int),
        "Error":       (m["test"].values - m["pred_mean"].values).astype(int),
        "APE (%)":     np.round(np.abs((m["test"].values - m["pred_mean"].values)
                                       / m["test"].values) * 100, 2),
    })
    df_outlier = pd.DataFrame({
        "Periode":       [d.strftime("%Y-%m") for d in outlier_info["dates"]],
        "Bulan":         [d.strftime("%B %Y")  for d in outlier_info["dates"]],
        "Nilai_Asli":    [int(ts_raw[d])        for d in outlier_info["dates"]],
        "Nilai_Imputed": [int(ts[d])            for d in outlier_info["dates"]],
    }) if outlier_info["dates"] else pd.DataFrame()

    df_summary = pd.DataFrame({
        "Parameter": ["Model","Order (p,d,q)","Seasonal Order","AIC","BIC","MAE","RMSE","MAPE (%)","Akurasi (%)","Pearson R","R-squared"],
        "Nilai": [
            f"SARIMA{m['order']}×{m['seasonal_order'][:3]}[s=12]",
            str(m["order"]), str(m["seasonal_order"]),
            round(m["aic"], 2), round(m["bic"], 2),
            round(m["mae"], 0), round(m["rmse"], 0),
            round(m["mape"], 2), round(m["acc"], 2),
            round(m["r"], 4), round(m["r"]**2, 4),
        ]
    })

    col1, col2 = st.columns(2)
    with col1:
        sec_header("📊", "Preview Prediksi 12 Bulan")
        df_fc_disp = df_forecast.copy()
        for col in ["Prediksi", "Lower_95CI", "Upper_95CI"]:
            df_fc_disp[col] = df_fc_disp[col].apply(fmt_num)
        st.dataframe(df_fc_disp, use_container_width=True, hide_index=True)

        # CSV download
        csv_fc = df_forecast.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ Download Prediksi (CSV)", csv_fc,
                           "prediksi_kedepan.csv", "text/csv", use_container_width=True)

    with col2:
        sec_header("✅", "Preview Evaluasi Test")
        df_eval_disp = df_eval.copy()
        for col in ["Aktual", "Prediksi", "CI_Bawah_95", "CI_Atas_95", "Error"]:
            df_eval_disp[col] = df_eval_disp[col].apply(fmt_num)
        st.dataframe(df_eval_disp, use_container_width=True, hide_index=True)

        csv_ev = df_eval.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ Download Evaluasi (CSV)", csv_ev,
                           "evaluasi_test.csv", "text/csv", use_container_width=True)

    # Excel download
    sec_header("📑", "Download Excel Lengkap (5 Sheet)")
    excel_buf = io.BytesIO()
    df_hist_dl = df_raw[["Periode","Total_Penumpang","Penumpang_Datang","Penumpang_Berangkat",
                          "Total_Pesawat","is_outlier"]].copy()
    df_hist_dl["Periode"] = df_hist_dl["Periode"].dt.strftime("%Y-%m")
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df_hist_dl.to_excel(writer, sheet_name="Data_Historis",   index=False)
        df_forecast.to_excel(writer, sheet_name="Prediksi_12_Bln", index=False)
        df_eval.to_excel(writer,     sheet_name="Evaluasi_Test",   index=False)
        if not df_outlier.empty:
            df_outlier.to_excel(writer, sheet_name="Outlier_Log",  index=False)
        df_summary.to_excel(writer,  sheet_name="Ringkasan_Model", index=False)
    excel_buf.seek(0)

    st.markdown("""
<div class="box-info">
📑 File Excel berisi <b>5 sheet</b>: Data Historis, Prediksi 12 Bulan,
Evaluasi Test, Outlier Log, dan Ringkasan Model.
</div>""", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download Excel Lengkap (5 Sheet)",
        excel_buf.getvalue(),
        "hasil_fusion_sarima.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=False,
    )

    sec_header("📋", "Ringkasan Model")
    st.dataframe(df_summary, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div class="sb-brand">
  <div class="sb-brand-icon">✈️</div>
  <h2>FusionSARIMA-UPG</h2>
  <p>Bandara Sultan Hasanuddin<br>Makassar · Penumpang Domestik</p>
</div>""", unsafe_allow_html=True)

        # Upload atau pakai file default
        st.markdown("### 📂 Sumber Data")
        uploaded = st.file_uploader("Upload CSV (opsional)", type=["csv"],
                                    help="Biarkan kosong untuk pakai data default (lalinud_UPG_domestik.csv)")

        default_csv = os.path.join(BASE_DIR, "lalinud_UPG_domestik.csv")
        if uploaded:
            file_bytes = uploaded.read()
        else:
            if not os.path.exists(default_csv):
                candidates = (
                    glob.glob(os.path.join(BASE_DIR, "*UPG*domestik*.csv")) +
                    glob.glob(os.path.join(BASE_DIR, "*.csv"))
                )
                if candidates:
                    default_csv = candidates[0]
            with open(default_csv, "rb") as f:
                file_bytes = f.read()

        # Cache data in session state
        if "file_bytes" not in st.session_state or st.session_state["file_bytes"] != file_bytes:
            st.session_state["file_bytes"]    = file_bytes
            st.session_state["model_result"]  = None
            st.session_state["outlier_config"] = (3.5, 1.5)
            with st.spinner("Memuat data..."):
                st.session_state["data"] = load_and_preprocess(file_bytes)

        # Data info
        if "data" in st.session_state:
            _, _, ts, oi = st.session_state["data"]
            st.markdown(f"""
<div class="box-info" style="font-size:12px;margin-top:8px;">
📅 <b>{ts.index[0].strftime('%b %Y')}</b> — <b>{ts.index[-1].strftime('%b %Y')}</b><br>
📊 <b>{len(ts)}</b> observasi · <b>{len(oi['dates'])}</b> outlier
</div>""", unsafe_allow_html=True)

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)

        # Navigation
        st.markdown("### 🗺️ Navigasi")
        pages = {
            "🏠 Dashboard":                 "dashboard",
            "📊 Eksplorasi Data (EDA)":     "eda",
            "🔍 Deteksi Outlier":           "outlier",
            "📈 Stasioneritas & Dekomposisi": "stationarity",
            "🤖 Model & Prediksi":          "model",
            "🧪 Diagnostik Residual":       "diagnostics",
            "📥 Download Hasil":            "download",
        }
        selected = st.radio("Pilih Halaman", list(pages.keys()), label_visibility="collapsed")

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)

        # Model status in sidebar
        m = st.session_state.get("model_result")
        if m:
            st.markdown(f"""
<div class="box-success" style="font-size:12px;">
✅ <b>Model Aktif</b><br>
SARIMA{m['order']}×{m['seasonal_order'][:3]}<br>
MAPE: {m['mape']:.2f}% · Akurasi: {m['acc']:.2f}%
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="box-warn" style="font-size:12px;">
⚠️ Model belum dijalankan.<br>
Buka <b>🤖 Model &amp; Prediksi</b> untuk mulai.
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div style="text-align:center;color:#334155;font-size:11px;margin-top:24px;">
  FusionSARIMA-UPG © 2026<br>Skripsi · Universitas Persatuan Guru
</div>""", unsafe_allow_html=True)

        return pages[selected]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    # Initialize session state
    if "model_result" not in st.session_state:
        st.session_state["model_result"]   = None
    if "outlier_config" not in st.session_state:
        st.session_state["outlier_config"] = (3.5, 1.5)

    page = render_sidebar()

    if "data" not in st.session_state:
        st.warning("⚠️ Tidak ada data. Pastikan file CSV tersedia di folder yang sama dengan app.py.")
        return

    if   page == "dashboard":    page_dashboard()
    elif page == "eda":          page_eda()
    elif page == "outlier":      page_outlier()
    elif page == "stationarity": page_stationarity()
    elif page == "model":        page_model()
    elif page == "diagnostics":  page_diagnostics()
    elif page == "download":     page_download()


if __name__ == "__main__":
    main()
