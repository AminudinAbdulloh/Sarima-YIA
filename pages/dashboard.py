# -*- coding: utf-8 -*-
"""pages/dashboard.py — Dashboard overview page."""
import streamlit as st
import pandas as pd
from components import (
    kpi_card, kpi_grid, section_header, page_hero,
    fmt, fmt_num, fmt_dec, fmt_pct, app_footer,
)
from chart_utils import chart_ts_overview, chart_seasonal_pattern


def render() -> None:
    df_raw, _, ts = st.session_state["data"]
    m = st.session_state.get("model_result")

    # ── Hero ────────────────────────────────────────────────────────────────
    page_hero(
        "Metode SARIMA · Domestik",
        "Prediksi Penumpang Bandara YIA",
        "Analisis deskriptif dan proyeksi penumpang domestik "
        "menggunakan model Seasonal ARIMA.",
    )

    # ── KPI cards ───────────────────────────────────────────────────────────
    if m:
        kpi_grid(
            kpi_card("Total Data Historis",
                     fmt(ts.sum()), f"{len(ts)} bulan observasi", "🗂️", "blue"),
            kpi_card("Prediksi ke Depan",
                     fmt(sum(r["Prediksi"] for r in m["forecast_rows"])),
                     f"Horizon {len(m['forecast_rows'])} bulan", "🔮", "green"),
            kpi_card("Akurasi Model",
                     fmt_pct(m["acc"]), f"MAPE: {m['mape']:.2f}%", "🎯", "amber"),
            kpi_card("Pearson R²",
                     f"{m['r'] ** 2:.4f}", f"R = {m['r']:.4f}", "📐", "purple"),
        )
    else:
        kpi_grid(
            kpi_card("Total Data Historis",
                     fmt(ts.sum()), f"{len(ts)} bulan observasi", "🗂️", "blue"),
            kpi_card("Periode Data",
                     ts.index[0].strftime("%b %Y"),
                     f"s/d {ts.index[-1].strftime('%b %Y')}", "📅", "amber"),
            kpi_card("Rata-rata / Bulan",
                     fmt(ts.mean()), "Penumpang per bulan", "📊", "purple"),
            kpi_card("Status Model",
                     "Siap",
                     "Buka halaman Model & Prediksi", "⚡", "green"),
        )

    # ── Charts row ──────────────────────────────────────────────────────────
    col_ts, col_sea = st.columns([3, 2], gap="medium")
    with col_ts:
        section_header(
            "Deret Waktu",
            "Total Penumpang Domestik per Bulan",
            f"{ts.index[0].strftime('%B %Y')} – {ts.index[-1].strftime('%B %Y')}",
        )
        st.plotly_chart(chart_ts_overview(ts), use_container_width=True)

    with col_sea:
        section_header(
            "Musiman",
            "Pola Bulanan",
            "Rata-rata per bulan (ribu penumpang)",
        )
        st.plotly_chart(chart_seasonal_pattern(ts), use_container_width=True)

    # ── Yearly summary table ─────────────────────────────────────────────────
    section_header(
        "Agregasi",
        "Ringkasan Tahunan",
        "Total, rata-rata, minimum, dan maksimum per tahun",
    )
    yearly = (
        df_raw.groupby("Tahun")["Total_Penumpang"]
        .agg(Total="sum", **{"Rata-rata / Bulan": "mean"}, Minimum="min", Maksimum="max")
        .reset_index()
    )
    for col in ["Total", "Rata-rata / Bulan", "Minimum", "Maksimum"]:
        yearly[col] = yearly[col].apply(fmt_num)
    yearly["Tahun"] = yearly["Tahun"].astype(int).astype(str)
    st.dataframe(yearly, use_container_width=True, hide_index=True)

    # ── Forecast preview ─────────────────────────────────────────────────────
    if m:
        section_header(
            "Prediksi ke Depan",
            f"Hasil Forecast SARIMA{m['order']}×{m['seasonal_order'][:3]}[s=12]",
            f"Horizon {len(m['forecast_rows'])} bulan",
        )
        df_fc = pd.DataFrame(m["forecast_rows"]).copy()
        df_fc["Prediksi"] = df_fc["Prediksi"].apply(fmt_num)
        st.dataframe(
            df_fc[["Bulan", "Prediksi"]],
            use_container_width=True, hide_index=True,
        )

    # ── Footer ───────────────────────────────────────────────────────────────
    app_footer(
        "LalinudSARIMA · Prediksi Penumpang Domestik Bandara YIA · "
        f"Data periode {ts.index[0].strftime('%Y')}–{ts.index[-1].strftime('%Y')}."
    )
