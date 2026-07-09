# -*- coding: utf-8 -*-
"""pages/stationarity.py — Stationarity tests, decomposition, ACF/PACF."""
import pandas as pd
import streamlit as st
from components import section_header, page_hero, info_box
from chart_utils import chart_decomp, chart_acf_pacf_matplotlib
from data_utils import run_stationarity, run_decomposition
import matplotlib.pyplot as plt


def render() -> None:
    _, _, ts = st.session_state["data"]
    ts_bytes  = ts.to_json().encode()

    page_hero(
        "Analisis Stasioneritas",
        "Stasioneritas & Dekomposisi",
        "Uji ADF, KPSS, dekomposisi time series, serta plot ACF & PACF.",
    )

    tabs = st.tabs(["🧮 ADF + KPSS", "📉 Dekomposisi", "📊 ACF & PACF"])

    # ── Tab 1: Stationarity tests ─────────────────────────────────────────────
    with tabs[0]:
        section_header("Uji Statistik", "Hasil Uji Stasioneritas",
                       "ADF Test (H₀: tidak stasioner) · KPSS Test (H₀: stasioner)")

        with st.spinner("Menghitung uji stasioneritas…"):
            stat_res = run_stationarity(ts_bytes)

        rows = []
        for label, r in stat_res.items():
            rows.append({
                "Series":       label,
                "ADF Stat":     f"{r['adf_stat']:.4f}".replace(".", ","),
                "ADF p-value":  f"{r['adf_pval']:.4f}".replace(".", ","),
                "ADF Hasil":    "✅ Stasioner" if r["adf_ok"] else "❌ Tidak Stasioner",
                "KPSS Stat":    f"{r['kpss_stat']:.4f}".replace(".", ","),
                "KPSS p-value": f"{r['kpss_pval']:.4f}".replace(".", ","),
                "KPSS Hasil":   "✅ Stasioner" if r["kpss_ok"] else "❌ Tidak Stasioner",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        info_box(
            "💡 <b>Interpretasi:</b> ADF tolak H₀ jika p ≤ 0.05 → stasioner. "
            "KPSS tolak H₀ jika p ≤ 0.05 → tidak stasioner. "
            "Keduanya harus sepakat untuk konfirmasi.",
            "info",
        )

        st.markdown("---")
        for label, r in stat_res.items():
            with st.expander(f"📋 Detail — {label}", expanded=(label == "Original")):
                c1, c2 = st.columns(2)
                c1.markdown(f"""**ADF Test**
- Statistik: `{r['adf_stat']:.4f}`
- p-value: `{r['adf_pval']:.4f}`
- **{'✅ STASIONER (p ≤ 0,05)' if r['adf_ok'] else '❌ TIDAK STASIONER (p > 0,05)'}**""")
                c2.markdown(f"""**KPSS Test**
- Statistik: `{r['kpss_stat']:.4f}`
- p-value: `{r['kpss_pval']:.4f}`
- **{'✅ STASIONER (p > 0,05)' if r['kpss_ok'] else '❌ TIDAK STASIONER (p ≤ 0,05)'}**""")

    # ── Tab 2: Decomposition ──────────────────────────────────────────────────
    with tabs[1]:
        section_header("Dekomposisi", "Dekomposisi Time Series Aditif",
                       "Komponen: Trend · Musiman (s = 12) · Residual")
        with st.spinner("Menghitung dekomposisi…"):
            decomp_dict = run_decomposition(ts_bytes)
        st.plotly_chart(chart_decomp(decomp_dict), use_container_width=True)
        info_box(
            "💡 <b>Model Aditif:</b> Nilai = Trend + Seasonal + Residual. "
            "Sesuai untuk data dengan amplitudo musiman yang relatif konstan.",
            "info",
        )

    # ── Tab 3: ACF & PACF ─────────────────────────────────────────────────────
    with tabs[2]:
        section_header("Korelogram", "ACF & PACF",
                       "Data asli dan setelah differencing (d=1, D=1)")
        with st.spinner("Membuat plot ACF & PACF…"):
            fig_acf = chart_acf_pacf_matplotlib(ts)
        st.pyplot(fig_acf, use_container_width=True)
        plt.close("all")
        info_box(
            "💡 <b>Panduan:</b> ACF → orde MA (q, Q) · "
            "PACF → orde AR (p, P). "
            "Garis putus-putus = batas kepercayaan 95%.",
            "info",
        )
