# -*- coding: utf-8 -*-
"""
config.py — Design tokens, constants, and application-level configuration.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Locale helpers ─────────────────────────────────────────────────────────────
BULAN_FULL = {
    1: "Januari",  2: "Februari", 3: "Maret",    4: "April",
    5: "Mei",      6: "Juni",     7: "Juli",      8: "Agustus",
    9: "September",10: "Oktober", 11: "November", 12: "Desember",
}
BULAN_SHORT = {k: v[:3] for k, v in BULAN_FULL.items()}

# ── Design tokens ──────────────────────────────────────────────────────────────
C = {
    # Primary blues
    "blue":      "#2563EB",
    "blue_md":   "#3B82F6",
    "blue_lt":   "#EFF6FF",
    "blue_fill": "rgba(37,99,235,0.07)",
    # Semantic
    "green":     "#059669",
    "green_lt":  "#ECFDF5",
    "amber":     "#D97706",
    "amber_lt":  "#FFFBEB",
    "red":       "#DC2626",
    "red_lt":    "#FEF2F2",
    "purple":    "#7C3AED",
    "purple_lt": "#F5F3FF",
    "cyan":      "#0891B2",
    "cyan_lt":   "#ECFEFF",
    # Neutrals
    "bg_page":   "#F1F5F9",
    "bg_card":   "#FFFFFF",
    "border":    "#E2E8F0",
    "border_md": "#CBD5E1",
    "text":      "#0F172A",
    "body":      "#374151",
    "muted":     "#6B7280",
    "subtle":    "#94A3B8",
    "label":     "#2563EB",
}

# ── Plotly chart theme ─────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#374151", family="Inter, sans-serif", size=11),
    margin=dict(l=40, r=20, t=44, b=40),
    legend=dict(
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#E2E8F0",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        zeroline=False,
        tickfont=dict(color="#6B7280", size=10),
    ),
    yaxis=dict(
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        zeroline=False,
        tickfont=dict(color="#6B7280", size=10),
    ),
)

# ── Navigation ─────────────────────────────────────────────────────────────────
PAGES = {
    "🏠 Dashboard":      "dashboard",
    "🤖 Metode SARIMA":  "model",
    "📥 Download Hasil": "download",
}


# ── Model constraints (sesuai batasan masalah) ─────────────────────────────────
AUTO_ARIMA_PARAMS = dict(
    start_p=0, max_p=3, start_q=0, max_q=3,
    d=None,    max_d=2,
    start_P=0, max_P=2, start_Q=0, max_Q=2,
    D=1,       m=12,
    seasonal=True,
    information_criterion="aic",
    stepwise=True,
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
)
TEST_SIZE = 12  # fixed — batasan masalah no. 5
