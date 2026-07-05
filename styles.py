# -*- coding: utf-8 -*-
"""
styles.py — Custom CSS injection for the light-theme dashboard.
Matches the reference design: clean white cards, blue accents, Inter typography.
"""
import streamlit as st

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900&display=swap');

/* ── Base & reset ─────────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="block-container"] { padding: 2rem 2.5rem 3rem; }
#MainMenu, footer               { visibility: hidden; }
[data-testid="stDecoration"]    { display: none; }
[data-testid="stSidebarNav"]    { display: none; }   /* hide auto pages/ nav */
header                          { background: transparent !important; }

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio > div   { gap: 2px; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    border-radius: 8px !important;
    padding: 9px 12px !important;
    transition: background 0.15s ease, color 0.15s ease !important;
    cursor: pointer; width: 100%;
    color: #374151 !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: #EFF6FF !important;
    color: #2563EB !important;
}

/* ── KPI Grid & Cards ────────────────────────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 22px 18px;
    position: relative;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.kpi-card:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.07);
    transform: translateY(-2px);
}
.kpi-icon-wrap {
    position: absolute; top: 18px; right: 18px;
    width: 38px; height: 38px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.kpi-icon-wrap.blue   { background: #EFF6FF; }
.kpi-icon-wrap.green  { background: #ECFDF5; }
.kpi-icon-wrap.amber  { background: #FFFBEB; }
.kpi-icon-wrap.purple { background: #F5F3FF; }
.kpi-icon-wrap.red    { background: #FEF2F2; }
.kpi-icon-wrap.cyan   { background: #ECFEFF; }

.kpi-label {
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em;
    color: #6B7280; margin-bottom: 10px;
}
.kpi-value { font-size: 30px; font-weight: 800; color: #0F172A; line-height: 1.1; }
.kpi-sub   { font-size: 12px; color: #6B7280; margin-top: 6px; }

/* ── Section header ──────────────────────────────────────────────────── */
.sec-wrap { margin: 28px 0 14px; }
.sec-eyebrow {
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #2563EB; margin-bottom: 3px;
}
.sec-title { font-size: 20px; font-weight: 700; color: #0F172A; margin: 0 0 3px; }
.sec-sub   { font-size: 13px; color: #2563EB; }

/* ── Page hero ───────────────────────────────────────────────────────── */
.page-hero            { margin-bottom: 28px; }
.page-hero-eyebrow {
    font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #2563EB; margin-bottom: 6px;
}
.page-hero h1 {
    font-size: 34px; font-weight: 800; color: #0F172A;
    margin: 0 0 8px; line-height: 1.15;
}
.page-hero p { font-size: 14px; color: #374151; margin: 0; max-width: 560px; }

/* ── Info / alert boxes ──────────────────────────────────────────────── */
.box {
    border-radius: 8px; padding: 11px 16px;
    font-size: 13px; margin: 10px 0;
    border-left: 3px solid;
}
.box-info    { background: #EFF6FF; border-color: #2563EB; color: #1D4ED8; }
.box-success { background: #ECFDF5; border-color: #059669; color: #047857; }
.box-warn    { background: #FFFBEB; border-color: #D97706; color: #B45309; }
.box-err     { background: #FEF2F2; border-color: #DC2626; color: #B91C1C; }

/* ── Badge ───────────────────────────────────────────────────────────── */
.badge {
    display: inline-block; border-radius: 20px;
    padding: 2px 10px; font-size: 11px; font-weight: 700;
}
.badge-ok   { background: #ECFDF5; color: #059669; }
.badge-warn { background: #FFFBEB; color: #D97706; }
.badge-err  { background: #FEF2F2; color: #DC2626; }

/* ── Sidebar brand ───────────────────────────────────────────────────── */
.sb-brand { padding: 16px 4px 14px; margin-bottom: 2px; }
.sb-brand-eyebrow {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #94A3B8; margin-bottom: 3px;
}
.sb-brand h2 { font-size: 17px; font-weight: 800; color: #0F172A; margin: 0 0 2px; }
.sb-brand p  { font-size: 11px; color: #6B7280; margin: 0; line-height: 1.5; }

/* ── Divider ─────────────────────────────────────────────────────────── */
.divider { border: none; border-top: 1px solid #E2E8F0; margin: 14px 0; }

/* ── Streamlit widget overrides ──────────────────────────────────────── */
/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important; font-size: 13px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden; }
[data-testid="stDataFrame"] > div {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #F1F5F9; border-radius: 10px; padding: 4px; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px; color: #6B7280; font-weight: 600;
    font-size: 13px; padding: 6px 14px;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important; color: #0F172A !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 10px; padding: 14px 16px;
}
[data-testid="stMetricLabel"] {
    color: #6B7280 !important; font-size: 10px !important;
    font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stMetricValue"] {
    color: #0F172A !important; font-size: 22px !important;
    font-weight: 800 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #CBD5E1;
    border-radius: 10px; padding: 6px;
}

/* App footer */
.app-footer {
    margin-top: 48px; padding-top: 16px;
    border-top: 1px solid #E2E8F0;
    font-size: 11px; color: #94A3B8;
}
.app-footer a { color: #94A3B8; text-decoration: none; }
"""


def inject_css() -> None:
    """Inject all custom CSS into the running Streamlit app."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
