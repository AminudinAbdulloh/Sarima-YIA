# -*- coding: utf-8 -*-
"""
chart_utils.py — Plotly and Matplotlib chart builders for the dashboard.

Each function returns a figure object; rendering is handled by the caller.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf as acf_fn
from scipy import stats

from config import C, BULAN_FULL
from components import apply_plotly_theme


# ── Helper ─────────────────────────────────────────────────────────────────────

def _dict_to_series(d: dict) -> pd.Series:
    s = pd.Series(d)
    s.index = pd.to_datetime(s.index)
    return s


# ── EDA charts ─────────────────────────────────────────────────────────────────

def chart_ts_overview(ts: pd.Series) -> go.Figure:
    """Line chart of total monthly passengers."""
    fig = go.Figure(go.Scatter(
        x=ts.index, y=ts.values,
        name="Total Penumpang",
        line=dict(color=C["blue"], width=2.2),
        fill="tozeroy",
        fillcolor=C["blue_fill"],
    ))
    apply_plotly_theme(fig, height=380)
    fig.update_yaxes(tickformat=",.0f")
    return fig


def chart_datang_berangkat(df: pd.DataFrame) -> go.Figure:
    """Stacked area chart: arriving vs departing passengers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Periode"], y=df["Penumpang_Datang"],
        name="Penumpang Datang",
        line=dict(color=C["blue"], width=2),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=df["Periode"], y=df["Penumpang_Berangkat"],
        name="Penumpang Berangkat",
        line=dict(color=C["amber"], width=2),
        fill="tozeroy", fillcolor="rgba(215,119,6,0.07)",
    ))
    apply_plotly_theme(fig, height=360)
    fig.update_yaxes(tickformat=",.0f")
    return fig

def chart_seasonal_pattern(ts: pd.Series) -> go.Figure:
    """Bar chart of average monthly passengers to reveal seasonality."""
    monthly_avg = ts.groupby(ts.index.month).mean()
    overall_avg = monthly_avg.mean()
    bulan_names = [list(BULAN_FULL.values())[m - 1][:3] for m in monthly_avg.index]

    fig = go.Figure(go.Bar(
        x=bulan_names,
        y=monthly_avg.values,
        marker=dict(color=C["blue"], opacity=0.82, line=dict(width=0)),
        text=[f"{v / 1000:.0f}K" for v in monthly_avg.values],
        textposition="outside",
        textfont=dict(size=9, color="#6B7280"),
        name="Rata-rata",
    ))
    fig.add_hline(
        y=overall_avg,
        line=dict(color="#CBD5E1", dash="dash", width=1.5),
        annotation_text=f"Rata-rata: {overall_avg / 1000:.0f}K",
        annotation_font=dict(color="#6B7280", size=10),
    )
    apply_plotly_theme(fig, height=360)
    fig.update_yaxes(tickformat=",.0f")
    return fig


# ── Stationarity / decomposition charts ────────────────────────────────────────

def chart_decomp(decomp_dict: dict) -> go.Figure:
    """4-panel additive decomposition: observed, trend, seasonal, residual."""
    obs  = _dict_to_series(decomp_dict["observed"])
    trnd = _dict_to_series(decomp_dict["trend"])
    seas = _dict_to_series(decomp_dict["seasonal"])
    res  = _dict_to_series(decomp_dict["resid"])

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=["Data Asli", "Trend", "Musiman (Seasonal)", "Residual"],
        shared_xaxes=True, vertical_spacing=0.06,
    )
    palette = [C["blue"], C["amber"], C["green"], C["red"]]
    for i, (s, color) in enumerate(zip([obs, trnd, seas, res], palette), 1):
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            line=dict(color=color, width=1.8),
            name=fig.layout.annotations[i - 1].text,
            fill="tozeroy" if i == 1 else "none",
            fillcolor=C["blue_fill"] if i == 1 else None,
        ), row=i, col=1)
    fig.add_hline(y=0, line=dict(color="#E2E8F0", dash="dash", width=1), row=4, col=1)
    apply_plotly_theme(fig, "Dekomposisi Time Series Aditif (s = 12)", height=680)
    return fig


def chart_acf_pacf_matplotlib(ts: pd.Series) -> plt.Figure:
    """2×2 ACF/PACF plots for original and differenced series."""
    ts_diff    = ts.diff().diff(12).dropna()
    nlags_orig = min(30, len(ts) // 2 - 1)
    nlags_diff = min(30, len(ts_diff) // 2 - 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    fig.set_facecolor("#FFFFFF")

    _ax_style = dict(facecolor="#FFFFFF")
    for ax in axes.flat:
        ax.set(**_ax_style)
        ax.tick_params(colors="#6B7280", labelsize=9)
        ax.xaxis.label.set_color("#6B7280")
        ax.yaxis.label.set_color("#6B7280")
        ax.title.set_color("#0F172A")
        for sp in ax.spines.values():
            sp.set_color("#E2E8F0")
        ax.grid(color="#F1F5F9", linewidth=0.8)

    plot_acf(ts,       ax=axes[0, 0], alpha=0.05, lags=nlags_orig, color=C["blue"])
    axes[0, 0].set_title("ACF — Data Asli", color="#0F172A", fontsize=11, fontweight="bold")
    plot_pacf(ts,      ax=axes[0, 1], alpha=0.05, lags=nlags_orig, color=C["blue"])
    axes[0, 1].set_title("PACF — Data Asli", color="#0F172A", fontsize=11, fontweight="bold")
    plot_acf(ts_diff,  ax=axes[1, 0], alpha=0.05, lags=nlags_diff, color=C["green"])
    axes[1, 0].set_title("ACF — Setelah Diff (d=1, D=1)", color="#0F172A", fontsize=11, fontweight="bold")
    plot_pacf(ts_diff, ax=axes[1, 1], alpha=0.05, lags=nlags_diff, color=C["green"])
    axes[1, 1].set_title("PACF — Setelah Diff (d=1, D=1)", color="#0F172A", fontsize=11, fontweight="bold")

    for ax in axes.flat:
        ax.set_xlabel("Lag", color="#6B7280", fontsize=9)
    plt.tight_layout(pad=2.0)
    return fig


# ── Model & forecast charts ────────────────────────────────────────────────────

def chart_forecast(ts: pd.Series, m: dict) -> go.Figure:
    """Full forecast chart: train, fitted, test actual/predicted, future forecast."""
    fig = go.Figure()
    train, test = m["train"], m["test"]

    # Training data area
    fig.add_trace(go.Scatter(
        x=train.index, y=train.values, name="Data Latih",
        line=dict(color=C["blue"], width=2),
        fill="tozeroy", fillcolor=C["blue_fill"],
    ))
    # Fitted values (dotted overlay)
    fig.add_trace(go.Scatter(
        x=m["fitted_full"].index, y=m["fitted_full"].values, name="Fitted SARIMA",
        line=dict(color=C["cyan"], width=1.2, dash="dot"), opacity=0.65,
    ))
    # Test actual
    fig.add_trace(go.Scatter(
        x=test.index, y=test.values, name="Aktual (Uji)",
        line=dict(color="#374151", width=2.2),
        mode="lines+markers", marker=dict(size=6, color="#374151"),
    ))
    # Test prediction
    fig.add_trace(go.Scatter(
        x=m["pred_mean"].index, y=m["pred_mean"].values,
        name=f"Prediksi Uji (MAPE = {m['mape']:.2f}%)",
        line=dict(color=C["amber"], width=2.2),
        mode="lines+markers", marker=dict(size=6, color=C["amber"]),
    ))
    # Future prediction with labels
    fig.add_trace(go.Scatter(
        x=m["future_mean"].index, y=m["future_mean"].values,
        name="Prediksi ke Depan",
        line=dict(color=C["green"], width=2.5),
        mode="lines+markers+text",
        marker=dict(size=7, color=C["green"]),
        text=[f"{v / 1000:.0f}K" for v in m["future_mean"].values],
        textposition="top center",
        textfont=dict(size=9, color=C["green"]),
    ))
    # Vertical dividers
    fig.add_vline(
        x=test.index[0],
        line=dict(color=C["amber"], width=1, dash="dot"),
    )
    fig.add_vline(
        x=m["future_mean"].index[0],
        line=dict(color=C["green"], width=1.5, dash="dashdot"),
    )
    apply_plotly_theme(fig, "Historis, Evaluasi & Prediksi SARIMA", height=500)
    fig.update_yaxes(tickformat=",.0f")
    return fig


def chart_actual_vs_pred(test: pd.Series, pred_mean: pd.Series) -> go.Figure:
    """Side-by-side line and scatter comparison of actual vs predicted."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Aktual vs Prediksi (Line)", "Aktual vs Prediksi (Scatter)"],
    )
    # Line chart
    fig.add_trace(go.Scatter(
        x=test.index, y=test.values, name="Aktual",
        line=dict(color="#374151", width=2),
        mode="lines+markers", marker=dict(size=6),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pred_mean.index, y=pred_mean.values, name="Prediksi",
        line=dict(color=C["amber"], width=2, dash="dash"),
        mode="lines+markers", marker=dict(size=6, symbol="square"),
    ), row=1, col=1)
    # Scatter chart
    lims = [
        min(test.min(), pred_mean.min()) * 0.90,
        max(test.max(), pred_mean.max()) * 1.08,
    ]
    fig.add_trace(go.Scatter(
        x=test.values, y=pred_mean.values, mode="markers",
        marker=dict(color=C["blue"], size=9, opacity=0.80), name="Pasangan Nilai",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=lims, y=lims, name="Garis Ideal (45°)",
        line=dict(color=C["red"], dash="dash", width=1.5),
    ), row=1, col=2)
    apply_plotly_theme(fig, height=400)
    fig.update_xaxes(tickformat=",.0f", row=1, col=2)
    fig.update_yaxes(tickformat=",.0f", row=1, col=2)
    return fig


# ── Residual diagnostics chart ─────────────────────────────────────────────────

def chart_residual_diagnostics(residuals: pd.Series) -> go.Figure:
    """4-panel residual diagnostics: time plot, histogram, Q-Q, ACF."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Residual vs Waktu", "Distribusi Residual",
            "Q-Q Plot",          "ACF Residual",
        ],
    )
    # Residual vs time
    fig.add_trace(go.Scatter(
        x=residuals.index, y=residuals.values, name="Residual",
        line=dict(color=C["blue"], width=1.5),
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#CBD5E1", dash="dash"), row=1, col=1)

    # Histogram + normal overlay
    fig.add_trace(go.Histogram(
        x=residuals.values, nbinsx=15,
        marker=dict(color=C["blue"], opacity=0.70, line=dict(color="#FFFFFF", width=0.5)),
        name="Histogram",
    ), row=1, col=2)
    x_norm = np.linspace(residuals.min(), residuals.max(), 120)
    y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
    scale  = len(residuals) * (residuals.max() - residuals.min()) / 15
    fig.add_trace(go.Scatter(
        x=x_norm, y=y_norm * scale, name="Distribusi Normal",
        line=dict(color=C["red"], width=2),
    ), row=1, col=2)

    # Q-Q plot
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    fig.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers",
        marker=dict(color=C["blue"], size=6), name="Q-Q Data",
    ), row=2, col=1)
    x_line = np.array([min(osm), max(osm)])
    fig.add_trace(go.Scatter(
        x=x_line, y=slope * x_line + intercept,
        line=dict(color=C["amber"], width=2), name="Garis Normal",
    ), row=2, col=1)

    # ACF residual (bar)
    acf_vals = acf_fn(residuals, nlags=24, fft=True)
    conf     = 1.96 / np.sqrt(len(residuals))
    colors   = [C["green"] if abs(v) <= conf else C["red"] for v in acf_vals]
    fig.add_trace(go.Bar(
        x=list(range(len(acf_vals))), y=acf_vals,
        marker=dict(color=colors, line=dict(width=0)), name="ACF Residual",
    ), row=2, col=2)
    fig.add_hline(y= conf, line=dict(color="#CBD5E1", dash="dot"), row=2, col=2)
    fig.add_hline(y=-conf, line=dict(color="#CBD5E1", dash="dot"), row=2, col=2)
    fig.add_hline(y=0,     line=dict(color="#E2E8F0"),             row=2, col=2)

    apply_plotly_theme(fig, "Diagnostik Residual Model SARIMA", height=640)
    return fig
