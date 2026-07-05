# -*- coding: utf-8 -*-
"""pages/eda.py — Exploratory Data Analysis page."""
import streamlit as st
from components import section_header, page_hero, info_box, fmt_num
from chart_utils import (
    chart_ts_overview, chart_datang_berangkat, chart_pesawat, chart_seasonal_pattern,
)


def render() -> None:
    df_raw, _, ts = st.session_state["data"]

    page_hero(
        "Eksplorasi Data",
        "Analisis Deskriptif Data Penumpang",
        "Visualisasi dan statistik data historis lalu lintas udara domestik.",
    )

    # ── Summary metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Periode Awal",   ts.index[0].strftime("%B %Y"))
    c2.metric("Periode Akhir",  ts.index[-1].strftime("%B %Y"))
    c3.metric("Total Observasi", f"{len(ts)} bulan")
    c4.metric("Rata-rata/Bln",  fmt_num(ts.mean()))

    # ── Tabbed charts ────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Overview",
        "🛬🛫 Datang & Berangkat",
        "✈️ Pergerakan Pesawat",
        "📋 Data Mentah",
        "📊 Statistik Deskriptif",
    ])

    with tabs[0]:
        section_header("Deret Waktu", "Total Penumpang Domestik per Bulan",
                       f"{ts.index[0].strftime('%B %Y')} – {ts.index[-1].strftime('%B %Y')}")
        st.plotly_chart(chart_ts_overview(ts), use_container_width=True)

    with tabs[1]:
        section_header("Komponen", "Penumpang Datang vs Berangkat",
                       "Perbandingan arus penumpang masuk dan keluar")
        st.plotly_chart(chart_datang_berangkat(df_raw), use_container_width=True)

    with tabs[2]:
        info_box(
            "ℹ️ <b>Data Konteks:</b> Pergerakan pesawat ditampilkan sebagai "
            "informasi pendukung. Variabel yang dianalisis dan diprediksi adalah "
            "<b>total penumpang domestik</b> (datang + berangkat).",
            "info",
        )
        st.plotly_chart(chart_pesawat(df_raw), use_container_width=True)

    with tabs[3]:
        tahun_list = sorted(df_raw["Tahun"].unique(), reverse=True)
        sel_tahun  = st.multiselect(
            "Filter Tahun", tahun_list, default=tahun_list, key="eda_filter"
        )
        disp = df_raw[df_raw["Tahun"].isin(sel_tahun)][
            ["Periode", "Total_Penumpang", "Penumpang_Datang",
             "Penumpang_Berangkat", "Total_Pesawat"]
        ].copy()
        disp["Periode"] = disp["Periode"].dt.strftime("%Y-%m")
        st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

    with tabs[4]:
        info_box(
            "ℹ️ <b>Data Asli Digunakan:</b> Seluruh nilai historis dipertahankan "
            "tanpa penggantian melalui interpolasi.",
            "info",
        )
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Observasi",  f"{len(ts)} bulan")
        sc2.metric("Nilai Minimum",    fmt_num(ts.min()))
        sc3.metric("Nilai Maksimum",   fmt_num(ts.max()))
        sc4.metric("Rata-rata/Bulan",  fmt_num(ts.mean()))

        stats_data = {
            "Statistik": [
                "Total Keseluruhan", "Rata-rata/Bulan", "Median",
                "Std. Deviasi", "Nilai Minimum", "Nilai Maksimum",
                "Koefisien Variasi (CV)",
            ],
            "Nilai": [
                fmt_num(ts.sum()),   fmt_num(ts.mean()),  fmt_num(ts.median()),
                fmt_num(ts.std()),   fmt_num(ts.min()),   fmt_num(ts.max()),
                f"{ts.std() / ts.mean() * 100:.2f}%",
            ],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # ── Yearly aggregation table ─────────────────────────────────────────────
    section_header("Agregasi", "Statistik Tahunan", "Total penumpang per tahun")
    yearly = (
        df_raw.groupby("Tahun")["Total_Penumpang"]
        .agg(Total="sum", **{"Rata2/Bulan": "mean"}, Min="min", Max="max")
        .reset_index()
    )
    for col in ["Total", "Rata2/Bulan", "Min", "Max"]:
        yearly[col] = yearly[col].apply(lambda x: f"{x:,.0f}")
    st.dataframe(yearly, use_container_width=True, hide_index=True)
