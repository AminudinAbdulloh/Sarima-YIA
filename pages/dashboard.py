# -*- coding: utf-8 -*-
"""pages/dashboard.py — Dashboard overview and Exploratory Data Analysis (EDA)."""
import streamlit as st
import pandas as pd
from components import (
    kpi_card, kpi_grid, section_header, page_hero, info_box,
    fmt, fmt_num, fmt_dec, fmt_pct, app_footer,
)
from chart_utils import (
    chart_ts_overview, chart_seasonal_pattern, chart_datang_berangkat,
)


def render() -> None:
    df_raw, _, ts = st.session_state["data"]
    m = st.session_state.get("model_result")

    # ── Hero ────────────────────────────────────────────────────────────────
    page_hero(
        "Dashboard Utama & Eksplorasi Data",
        "Prediksi Penumpang Bandara YIA",
        "Analisis deskriptif, pola tren, musiman, dan proyeksi penumpang domestik menggunakan model Seasonal ARIMA.",
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
                     "Silakan buka menu Metode SARIMA", "⚡", "green"),
        )

    # ── Trend and Seasonality Charts ──────────────────────────────────────────
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

    # ── Highlight Trend & Seasonality (Requested Analysis) ────────────────────
    info_box(
        "💡 <b>Hasil Analisis Pola Data (Eksplorasi Data - EDA):</b><br>"
        "1. 📈 <b>Tren Naik (Upward Trend):</b> Data historis menunjukkan bahwa arus penumpang domestik di Bandara YIA secara konsisten mengalami tren pergerakan naik (growth trend) dari tahun 2021 hingga akhir tahun 2025.<br>"
        "2. 📅 <b>Variasi Musiman (Seasonal Variation):</b> Pola bulanan rata-rata menegaskan adanya pola musiman berulang tahunan (periode musiman <i>s = 12</i>). Puncak pergerakan penumpang terjadi pada musim liburan tengah tahun (Juni–Juli) dan liburan akhir tahun (Desember).",
        "info",
    )

    # ── Advanced EDA Section ──────────────────────────────────────────────────
    section_header(
        "Eksplorasi Data",
        "Karakteristik & Detail Data Historis",
        "Statistik deskriptif, perbandingan arus datang/berangkat, dan data mentah",
    )

    tabs = st.tabs([
        "🛬🛫 Arus Datang vs Berangkat",
        "📋 Tabel Data Mentah",
        "📊 Statistik Deskriptif",
    ])

    with tabs[0]:
        st.plotly_chart(chart_datang_berangkat(df_raw), use_container_width=True)

    with tabs[1]:
        tahun_list = sorted(df_raw["Tahun"].unique(), reverse=True)
        sel_tahun = st.multiselect(
            "Filter Tahun", tahun_list, default=tahun_list, key="dashboard_eda_filter"
        )
        disp = df_raw[df_raw["Tahun"].isin(sel_tahun)].copy()
        disp["Periode"] = disp["Periode"].dt.strftime("%Y-%m")
        
        def format_indo(val):
            if pd.isna(val):
                return ""
            try:
                return fmt_num(val)
            except Exception:
                return str(val)
                
        disp["Penumpang Datang"] = disp["Penumpang_Datang"].apply(format_indo)
        disp["Penumpang Berangkat"] = disp["Penumpang_Berangkat"].apply(format_indo)
        disp["Total Penumpang"] = disp["Total_Penumpang"].apply(format_indo)
        
        disp_show = disp[["Periode", "Penumpang Datang", "Penumpang Berangkat", "Total Penumpang"]]
        st.dataframe(disp_show.reset_index(drop=True), use_container_width=True, hide_index=True)

    with tabs[2]:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Observasi", f"{len(ts)} bulan")
        sc2.metric("Nilai Minimum", fmt_num(ts.min()))
        sc3.metric("Nilai Maksimum", fmt_num(ts.max()))
        sc4.metric("Rata-rata/Bulan", fmt_num(ts.mean()))

        stats_data = {
            "Statistik": [
                "Total Keseluruhan", "Rata-rata/Bulan", "Median",
                "Std. Deviasi", "Nilai Minimum", "Nilai Maksimum",
                "Koefisien Variasi (CV)",
            ],
            "Nilai": [
                fmt_num(ts.sum()), fmt_num(ts.mean()), fmt_num(ts.median()),
                fmt_num(ts.std()), fmt_num(ts.min()), fmt_num(ts.max()),
                f"{ts.std() / ts.mean() * 100:.2f}%",
            ],
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

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

