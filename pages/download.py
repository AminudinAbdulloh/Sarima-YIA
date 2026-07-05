# -*- coding: utf-8 -*-
"""pages/download.py — Export results as CSV and Excel."""
import io
import numpy as np
import pandas as pd
import streamlit as st
from components import section_header, page_hero, info_box, fmt_num


def render() -> None:
    m   = st.session_state.get("model_result")
    res = st.session_state.get("data")

    page_hero(
        "Export Hasil",
        "Download Hasil Analisis",
        "Unduh prediksi, evaluasi model, dan data historis dalam format CSV atau Excel.",
    )

    if not m:
        info_box(
            "⚠️ Model belum dijalankan. "
            "Buka halaman <b>🤖 Model & Prediksi</b> terlebih dahulu.",
            "warn",
        )
        return

    df_raw, _, ts = res

    # ── Prepare export dataframes ────────────────────────────────────────────
    df_forecast = pd.DataFrame(m["forecast_rows"])

    df_eval = pd.DataFrame({
        "Periode":     m["test"].index.strftime("%Y-%m"),
        "Aktual":      m["test"].values.astype(int),
        "Prediksi":    m["pred_mean"].values.astype(int),
        "CI_Bawah_95": m["pred_ci"].iloc[:, 0].values.astype(int),
        "CI_Atas_95":  m["pred_ci"].iloc[:, 1].values.astype(int),
        "Error":       (m["test"].values - m["pred_mean"].values).astype(int),
        "APE (%)":     np.round(
            np.abs((m["test"].values - m["pred_mean"].values) / m["test"].values) * 100, 2
        ),
    })

    df_summary = pd.DataFrame({
        "Parameter": [
            "Model", "Order (p,d,q)", "Seasonal Order",
            "AIC", "BIC", "MAE", "RMSE", "MAPE (%)", "Akurasi (%)", "Pearson R", "R-squared",
        ],
        "Nilai": [
            f"SARIMA{m['order']}×{m['seasonal_order'][:3]}[s=12]",
            str(m["order"]),
            str(m["seasonal_order"]),
            str(round(m["aic"],  2)),
            str(round(m["bic"],  2)),
            str(round(m["mae"],  0)),
            str(round(m["rmse"], 0)),
            str(round(m["mape"], 2)),
            str(round(m["acc"],  2)),
            str(round(m["r"],    4)),
            str(round(m["r"] ** 2, 4)),
        ],
    })

    # ── Preview & per-file downloads ─────────────────────────────────────────
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        section_header("Prediksi", f"Prediksi {len(m['forecast_rows'])} Bulan ke Depan")
        disp_fc = df_forecast.copy()
        for c in ["Prediksi", "Lower_95CI", "Upper_95CI"]:
            disp_fc[c] = disp_fc[c].apply(fmt_num)
        st.dataframe(disp_fc, use_container_width=True, hide_index=True)
        csv_fc = df_forecast.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ Download Prediksi (CSV)", csv_fc,
            "prediksi_kedepan.csv", "text/csv", use_container_width=True,
        )

    with col2:
        section_header("Evaluasi", "Perbandingan Aktual vs Prediksi (12 Bulan Uji)")
        disp_ev = df_eval.copy()
        for c in ["Aktual", "Prediksi", "CI_Bawah_95", "CI_Atas_95", "Error"]:
            disp_ev[c] = disp_ev[c].apply(fmt_num)
        st.dataframe(disp_ev, use_container_width=True, hide_index=True)
        csv_ev = df_eval.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ Download Evaluasi (CSV)", csv_ev,
            "evaluasi_test.csv", "text/csv", use_container_width=True,
        )

    # ── Excel bundle ─────────────────────────────────────────────────────────
    section_header("Bundle", "Download Excel Lengkap (4 Sheet)")
    info_box(
        "📑 File Excel berisi <b>4 sheet</b>: "
        "Data Historis · Prediksi ke Depan · Evaluasi Test · Ringkasan Model.",
        "info",
    )

    df_hist = df_raw[
        ["Periode", "Total_Penumpang", "Penumpang_Datang",
         "Penumpang_Berangkat", "Total_Pesawat"]
    ].copy()
    df_hist["Periode"] = df_hist["Periode"].dt.strftime("%Y-%m")

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df_hist.to_excel(writer,     sheet_name="Data_Historis",   index=False)
        df_forecast.to_excel(writer, sheet_name="Prediksi_Kedepan",index=False)
        df_eval.to_excel(writer,     sheet_name="Evaluasi_Test",    index=False)
        df_summary.to_excel(writer,  sheet_name="Ringkasan_Model",  index=False)
    excel_buf.seek(0)

    st.download_button(
        "⬇️ Download Excel Lengkap (4 Sheet)",
        excel_buf.getvalue(),
        "hasil_sarima_yia.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ── Model summary table ───────────────────────────────────────────────────
    section_header("Ringkasan", "Parameter & Metrik Model")
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
