# -*- coding: utf-8 -*-
"""
components.py — Reusable UI component helpers for the dashboard.

All HTML-rendering helpers live here so page modules stay clean.
"""
import streamlit as st
import plotly.graph_objects as go
from config import PLOTLY_LAYOUT


# ── Number formatters ──────────────────────────────────────────────────────────

def fmt(n: float) -> str:
    """Short human-readable number: 1.2M, 750.3K, etc."""
    n = float(n)
    if abs(n) >= 1_000_000: return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:     return f"{n / 1_000:.1f}K"
    return f"{n:,.0f}"

def fmt_num(n) -> str:
    """Full comma-separated integer string."""
    return f"{float(n):,.0f}"

def fmt_pct(n) -> str:
    """Percentage with 2 decimal places."""
    return f"{float(n):.2f}%"


# ── HTML components ────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, sub: str, icon: str, color: str = "blue") -> str:
    """Return HTML string for a single KPI card (use inside kpi_grid)."""
    return f"""
<div class="kpi-card">
  <div class="kpi-icon-wrap {color}">{icon}</div>
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>"""


def kpi_grid(*cards: str) -> None:
    """Render up to 4 KPI cards in a responsive grid."""
    st.markdown(
        f'<div class="kpi-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def section_header(eyebrow: str, title: str, sub: str = "") -> None:
    """
    Render a section header with:
      - eyebrow: small blue uppercase label (e.g. 'DERET WAKTU')
      - title:   bold dark heading
      - sub:     optional blue subtitle
    """
    sub_html = f'<div class="sec-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
<div class="sec-wrap">
  <div class="sec-eyebrow">{eyebrow}</div>
  <div class="sec-title">{title}</div>
  {sub_html}
</div>""", unsafe_allow_html=True)


def page_hero(eyebrow: str, title: str, description: str = "") -> None:
    """Render the top-of-page hero block."""
    desc_html = f"<p>{description}</p>" if description else ""
    st.markdown(f"""
<div class="page-hero">
  <div class="page-hero-eyebrow">{eyebrow}</div>
  <h1>{title}</h1>
  {desc_html}
</div>""", unsafe_allow_html=True)


def info_box(text: str, kind: str = "info") -> None:
    """Render a coloured info/alert box. kind ∈ {info, success, warn, err}."""
    st.markdown(f'<div class="box box-{kind}">{text}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "ok") -> str:
    """Return inline HTML badge. kind ∈ {ok, warn, err}."""
    return f'<span class="badge badge-{kind}">{text}</span>'


def divider() -> None:
    """Render a subtle horizontal rule."""
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


def app_footer(text: str) -> None:
    """Render the page-level footer."""
    st.markdown(f'<div class="app-footer">{text}</div>', unsafe_allow_html=True)


# ── Plotly helpers ─────────────────────────────────────────────────────────────

def apply_plotly_theme(
    fig: go.Figure,
    title: str = "",
    height: int = 420,
) -> go.Figure:
    """Apply the shared light-theme Plotly layout to a figure."""
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=title,
            font=dict(size=13, color="#0F172A", weight=700),
            x=0.0, xanchor="left", pad=dict(l=4),
        ),
        height=height,
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#E2E8F0",
            font_color="#0F172A",
            font_size=12,
        ),
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridwidth=0.5)
    return fig
