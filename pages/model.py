# -*- coding: utf-8 -*-
"""pages/model.py — SARIMA model configuration, fitting, and forecast results."""
import pandas as pd
import streamlit as st
from components import (
    section_header, page_hero, info_box,
    kpi_card, kpi_grid, fmt_num, fmt_pct,
)
from chart_utils import chart_forecast, chart_actual_vs_pred
from model_utils import run_model


def render() -> None:
    _, _, ts = st.session_state["data"]

    page_hero(
        "Model SARIMA",
        "Model & Prediksi",
        "Auto ARIMA berbasis AIC · Evaluasi 12 bulan terakhir · Forecast hingga 24 bulan.",
    )

    # ── Configuration panel ──────────────────────────────────────────────────
    with st.expander("⚙️ Konfigurasi Model", expanded=True):
        c1, c2 = st.columns(2)
        use_auto   = c1.toggle("🪄 Auto ARIMA (Rekomendasi)", value=True)
        n_forecast = c2.slider("Horizon Prediksi (bulan)", 6, 24, 12)

        info_box(
            "📌 <b>Periode evaluasi ditetapkan 12 bulan terakhir</b> "
            "sesuai batasan masalah penelitian (BM no. 5).",
            "info",
        )

        if not use_auto:
            st.markdown(
                "**Parameter Manual SARIMA** — "
                "p, q ∈ {0–3} · P, Q ∈ {0–2} · d ∈ {0–2} · D = 1 (tetap)"
            )
            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            p = int(mc1.number_input("p", 0, 3, 1))
            d = int(mc2.number_input("d", 0, 2, 1))
            q = int(mc3.number_input("q", 0, 3, 1))
            P = int(mc4.number_input("P", 0, 2, 1))
            D = int(mc5.number_input("D (=1)", 1, 1, 1, disabled=True))
            Q = int(mc6.number_input("Q", 0, 2, 1))
            manual_order    = (p, d, q)
            manual_seasonal = (P, D, Q)
        else:
            manual_order    = (1, 1, 1)
            manual_seasonal = (1, 1, 1)

    # ── Run / reset buttons ──────────────────────────────────────────────────
    btn_col, rst_col = st.columns([1, 4])
    with btn_col:
        run_btn = st.button("🚀 Jalankan Model", type="primary", use_container_width=True)
    with rst_col:
        if st.session_state.get("model_result"):
            if st.button("🗑️ Reset Hasil"):
                st.session_state["model_result"] = None
                st.rerun()

    if run_btn:
        with st.spinner("⏳ Auto ARIMA sedang mencari parameter terbaik… (1–5 menit)"):
            try:
                result = run_model(ts, n_forecast, use_auto, manual_order, manual_seasonal)
                st.session_state["model_result"] = result
                st.success("✅ Model berhasil dijalankan!")
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Error: {exc}")
                return

    m = st.session_state.get("model_result")
    if not m:
        info_box(
            "⚠️ Model belum dijalankan. Klik <b>🚀 Jalankan Model</b> di atas untuk memulai.",
            "warn",
        )
        return

    # ── Selected parameters ──────────────────────────────────────────────────
    st.markdown("---")
    section_header("Parameter Terpilih", f"SARIMA{m['order']}×{m['seasonal_order'][:3]}[s=12]",
                   "Berdasarkan minimisasi nilai AIC")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Order (p,d,q)",      str(m["order"]))
    p2.metric("Seasonal (P,D,Q,s)", str(m["seasonal_order"]))
    p3.metric("AIC",                f"{m['aic']:.2f}".replace(".", ","))
    p4.metric("BIC",                f"{m['bic']:.2f}".replace(".", ","))

    # ── Evaluation metrics ───────────────────────────────────────────────────
    section_header("Evaluasi", "Kinerja Model pada Data Uji (12 Bulan Terakhir)",
                   "MAE · RMSE · MAPE sesuai batasan masalah no. 3")
    kpi_grid(
        kpi_card("MAE",     fmt_num(round(m["mae"])),  "Mean Absolute Error",        "📏", "blue"),
        kpi_card("RMSE",    fmt_num(round(m["rmse"])), "Root Mean Squared Error",    "📐", "purple"),
        kpi_card("MAPE",    fmt_pct(m["mape"]),         "Mean Absolute % Error",      "📉", "amber"),
        kpi_card("Akurasi", fmt_pct(m["acc"]),          f"Pearson R = {m['r']:.4f}".replace(".", ","), "🎯", "green"),
    )

    # ── Main forecast chart ──────────────────────────────────────────────────
    section_header("Visualisasi", "Historis, Evaluasi & Prediksi SARIMA")
    st.plotly_chart(chart_forecast(ts, m), use_container_width=True)

    # ── Actual vs predicted + forecast table ─────────────────────────────────
    chart_col, table_col = st.columns(2, gap="medium")
    with chart_col:
        section_header("Validasi", "Aktual vs Prediksi — Periode Uji")
        st.plotly_chart(chart_actual_vs_pred(m["test"], m["pred_mean"]), use_container_width=True)

    with table_col:
        section_header("Forecast", f"Tabel Prediksi {n_forecast} Bulan ke Depan")
        df_fc = pd.DataFrame(m["forecast_rows"]).copy()
        for c in ["Prediksi", "Lower_95CI", "Upper_95CI"]:
            df_fc[c] = df_fc[c].apply(fmt_num)
        st.dataframe(
            df_fc[["Bulan", "Prediksi", "Lower_95CI", "Upper_95CI"]],
            use_container_width=True, hide_index=True,
        )
        total_pred = sum(r["Prediksi"] for r in m["forecast_rows"])
        info_box(
            f"✅ <b>Total Prediksi {n_forecast} Bulan:</b> "
            f"{fmt_num(total_pred)} penumpang",
            "success",
        )
