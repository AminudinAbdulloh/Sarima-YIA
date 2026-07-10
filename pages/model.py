# -*- coding: utf-8 -*-
"""pages/model.py — Interactive SARIMA model building pipeline with ADF & KPSS and dynamic candidates."""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from components import (
    section_header, page_hero, info_box,
    kpi_card, kpi_grid, fmt_num, fmt_pct, divider, app_footer
)
from chart_utils import (
    chart_forecast, chart_actual_vs_pred, 
    chart_acf_pacf_matplotlib, chart_residual_diagnostics
)
from model_utils import run_model


def render() -> None:
    _, _, ts = st.session_state["data"]

    page_hero(
        "Metode SARIMA",
        "Pipeline Pemodelan & Prediksi",
        "Eksekusi pipeline pemodelan SARIMA berdasarkan batas wilayah ACF/PACF dan optimasi Auto ARIMA.",
    )

    # ── Configuration panel ──────────────────────────────────────────────────
    with st.expander("⚙️ Konfigurasi Peramalan", expanded=True):
        n_forecast = st.slider("Horizon Prediksi (bulan ke depan)", 6, 24, 12)
        info_box(
            "📌 <b>Alur Kerja Pemodelan:</b><br>"
            "1. <b>Uji Stasioneritas:</b> Menguji deret waktu asli dan diferensiasi menggunakan uji ADF dan KPSS.<br>"
            "2. <b>Identifikasi Model:</b> Menganalisis plot ACF & PACF untuk memahami data.<br>"
            "3. <b>Auto ARIMA:</b> Sistem mencari orde optimal secara otomatis, lalu mengevaluasi model tetangga di sekitarnya.",
            "info",
        )

    # ── Run / reset buttons ──────────────────────────────────────────────────
    btn_col, rst_col = st.columns([1, 4])
    with btn_col:
        run_btn = st.button("🚀 Jalankan Model", type="primary", use_container_width=True)
    with rst_col:
        if st.session_state.get("pipeline_result"):
            if st.button("🗑️ Reset Hasil Pemodelan"):
                st.session_state["pipeline_result"] = None
                st.session_state["model_result"] = None
                st.rerun()

    if run_btn:
        with st.spinner("⏳ Menjalankan model SARIMA & mengevaluasi kandidat model…"):
            try:
                pipeline_res = run_model(ts, n_forecast)
                st.session_state["pipeline_result"] = pipeline_res
                # Set winner model as default active model
                st.session_state["model_result"] = pipeline_res["candidates"][0]
                st.success("✅ Model SARIMA berhasil diselesaikan!")
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Terjadi kesalahan saat fitting model: {exc}")
                return

    p_res = st.session_state.get("pipeline_result")
    if not p_res:
        info_box(
            "💡 Silakan klik tombol <b>🚀 Jalankan Model</b> di atas untuk memulai "
            "analisis stasioneritas, estimasi parameter, dan peramalan.",
            "warn",
        )
        return

    # ── STEP-BY-STEP SEQUENTIAL PIPELINE DISPLAY ─────────────────────────────
    st.markdown("---")
    st.subheader("📋 Pipeline Metode SARIMA (Hasil Eksekusi)")

    # --- LANGKAH 1 & 2 ---
    with st.expander("1️⃣ Langkah 1 & 2: Uji Stasioneritas & Diferensiasi (ADF & KPSS)", expanded=True):
        st.markdown("##### Tabel Evaluasi Stasioneritas (ADF & KPSS):")
        st.markdown(
            "Menguji stasioneritas rata-rata (ADF Test, H₀: tidak stasioner / memiliki unit root) "
            "dan stasioneritas tren/varians (KPSS Test, H₀: stasioner) pada data asli dan setelah diferensiasi."
        )
        
        s_steps = p_res["stationarity_steps"]
        rows = []
        for label, key in [
            ("Data Asli (Original)", "Original"),
            ("Diferensiasi Non-Musiman (d=1, D=0)", "Diff_NonSeasonal"),
            ("Diferensiasi Musiman (d=0, D=1)", "Diff_Seasonal"),
            ("Diferensiasi Keduanya (d=1, D=1)", "Diff_Both"),
        ]:
            r = s_steps[key]
            rows.append({
                "Data Runtun Waktu": label,
                "ADF Stat": f"{r['adf_stat']:.4f}".replace(".", ","),
                "ADF p-value": f"{r['adf_pval']:.4f}".replace(".", ","),
                "ADF Hasil": "✅ Stasioner" if r["adf_ok"] else "❌ Tidak Stasioner",
                "KPSS Stat": f"{r['kpss_stat']:.4f}".replace(".", ","),
                "KPSS p-value": f"{r['kpss_pval']:.4f}".replace(".", ","),
                "KPSS Hasil": "✅ Stasioner" if r["kpss_ok"] else "❌ Tidak Stasioner",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        
        st.info(
            "💡 <b>Kriteria Stasioneritas:</b> Deret stasioner dicapai jika uji ADF menolak H₀ (p-value ≤ 0,05) "
            "dan uji KPSS gagal menolak H₀ (p-value > 0,05). Keduanya harus terpenuhi untuk konfirmasi stasioneritas penuh."
        )

    # --- LANGKAH 3 ---
    with st.expander("3️⃣ Langkah 3: Identifikasi Model Tentatif (ACF & PACF)", expanded=False):
        st.markdown(
            "Grafik ACF dan PACF digunakan untuk memahami data dan membatasi wilayah pencarian orde parameter model SARIMA."
        )
        with st.spinner("Membuat grafik ACF & PACF..."):
            fig_acf = chart_acf_pacf_matplotlib(ts)
            st.pyplot(fig_acf, use_container_width=True)
            plt.close("all")
            
        st.markdown(
            "• **ACF (Autocorrelation Function)** digunakan untuk membatasi orde Moving Average ($q, Q$).<br>"
            "• **PACF (Partial Autocorrelation Function)** digunakan untuk membatasi orde Autoregressive ($p, P$).",
            unsafe_allow_html=True
        )

    # --- LANGKAH 4 ---
    candidates = p_res["candidates"]
    with st.expander("4️⃣ Langkah 4: Estimasi Parameter Model", expanded=True):
        st.markdown(
            "Model terbaik diidentifikasi menggunakan `auto.arima()`. Seluruh model kandidat (termasuk model tetangga) "
            "diestimasi parameternya menggunakan metode Maximum Likelihood Estimation (MLE)."
        )
        
        # Display comparison of candidates
        st.markdown("##### Tabel Perbandingan Performa Model Kandidat:")
        comp_rows = []
        for cand in candidates:
            lb_ok = cand["lb_p"] > 0.05
            comp_rows.append({
                "Model": cand["name"],
                "AIC": f"{cand['aic']:.2f}".replace(".", ","),
                "BIC": f"{cand['bic']:.2f}".replace(".", ","),
                "MAPE": f"{cand['mape']:.2f}%".replace(".", ","),
                "Akurasi": f"{cand['acc']:.2f}%".replace(".", ","),
                "Ljung-Box (White Noise)": "✅ Lolos" if lb_ok else "⚠️ Perlu Cek",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        
        # Active model selection dropdown
        st.markdown("---")
        model_names = [cand["name"] for cand in candidates]
        active_m = st.session_state.get("model_result")
        default_idx = 0
        if active_m:
            try:
                default_idx = model_names.index(active_m["name"])
            except ValueError:
                default_idx = 0
                
        selected_model_name = st.selectbox(
            "🔎 Pilih Model Aktif untuk Ditampilkan Detailnya:", 
            model_names, 
            index=default_idx,
            help="Pilih model untuk mengamati diagnostik sisaan dan hasil peramalannya."
        )
        
        active_model = next(cand for cand in candidates if cand["name"] == selected_model_name)
        st.session_state["model_result"] = active_model
        
        # Display active model coefficients table
        st.markdown(f"##### Tabel Estimasi Parameter MLE untuk {active_model['name']}:")
        st.dataframe(active_model["df_coef"], use_container_width=True, hide_index=True)
        
        # Display combined parameter table matching the diagram
        st.markdown("##### Tabel Hasil Estimasi Parameter Semua Model (Ringkasan Gabungan):")
        df_combined = p_res["df_combined_params"].copy()
        df_combined["P-value"] = df_combined["P-value"].apply(lambda val: f"{val:.4f}".replace(".", ","))
        df_combined["Alpha"] = df_combined["Alpha"].apply(lambda val: f"{val:.2f}".replace(".", ","))
        st.dataframe(df_combined, use_container_width=True, hide_index=True)

    # --- LANGKAH 5 ---
    with st.expander("5️⃣ Langkah 5: Uji Diagnostik (Evaluasi Residual)", expanded=False):
        st.markdown(
            f"Evaluasi residual (sisaan) untuk model aktif **{active_model['name']}**."
        )
        lb_ok = active_model["lb_p"] > 0.05
        norm_ok = active_model["norm_p"] > 0.05
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Uji Ljung-Box (Autokorelasi)**")
            st.markdown(f"- p-value: `{active_model['lb_p']:.4f}`")
            if lb_ok:
                st.success("✅ Lolos (p > 0,05): Tidak ada autokorelasi, residual bersifat White Noise.")
            else:
                st.warning("⚠️ Perlu Cek (p ≤ 0,05): Terdapat autokorelasi pada residual.")
                
        with c2:
            st.markdown("**Uji Shapiro-Wilk (Normalitas)**")
            st.markdown(f"- p-value: `{active_model['norm_p']:.4f}`")
            if norm_ok:
                st.success("✅ Lolos (p > 0,05): Residual terdistribusi normal.")
            else:
                st.info("ℹ️ Info (p ≤ 0,05): Residual tidak terdistribusi normal.")
                
        st.plotly_chart(chart_residual_diagnostics(active_model["residuals"]), use_container_width=True)

        if lb_ok:
            info_box(
                f"✅ <b>Kesimpulan Diagnostik ({active_model['name']}):</b> Residual bersifat white noise. "
                "Model layak digunakan untuk forecasting.",
                "success"
            )
        else:
            info_box(
                f"⚠️ <b>Kesimpulan Diagnostik ({active_model['name']}):</b> Residual masih mendeteksi autokorelasi. "
                "Peramalan tetap dapat dilakukan, namun disarankan memilih model tetangga alternatif yang lolos uji diagnostik.",
                "warn"
            )

    # --- LANGKAH 6 ---
    with st.expander("6️⃣ Langkah 6: Peramalan (Forecasting)", expanded=True):
        st.markdown(
            f"Peramalan nilai di masa depan menggunakan model aktif **{active_model['name']}**."
        )
        
        # Test performance metrics
        section_header("Evaluasi", "Akurasi Model pada Data Uji (12 Bulan Uji)", "Mengukur keandalan peramalan model")
        kpi_grid(
            kpi_card("MAE", fmt_num(round(active_model["mae"])), "Mean Absolute Error", "📏", "blue"),
            kpi_card("RMSE", fmt_num(round(active_model["rmse"])), "Root Mean Squared Error", "📐", "purple"),
            kpi_card("MAPE", fmt_pct(active_model["mape"]), "Mean Absolute Percentage Error", "📉", "amber"),
            kpi_card("Akurasi", fmt_pct(active_model["acc"]), f"R = {active_model['r']:.4f}".replace(".", ","), "🎯", "green"),
        )
        
        # Main forecast chart
        section_header("Grafik Prediksi", "Historis & Proyeksi Penumpang Domestik YIA")
        st.plotly_chart(chart_forecast(ts, active_model), use_container_width=True)

        # Actual vs predicted side by side with forecast table
        col_val, col_tbl = st.columns(2, gap="medium")
        with col_val:
            section_header("Validasi", "Aktual vs Prediksi (Periode Uji)")
            st.plotly_chart(chart_actual_vs_pred(active_model["test"], active_model["pred_mean"]), use_container_width=True)
            
        with col_tbl:
            section_header("Tabel Proyeksi", f"Prediksi {n_forecast} Bulan ke Depan")
            df_fc = pd.DataFrame(active_model["forecast_rows"]).copy()
            df_fc["Prediksi"] = df_fc["Prediksi"].apply(fmt_num)
            st.dataframe(df_fc[["Bulan", "Prediksi"]], use_container_width=True, hide_index=True)
            
            total_pred = sum(r["Prediksi"] for r in active_model["forecast_rows"])
            info_box(
                f"🔮 <b>Total Prediksi Penumpang {n_forecast} Bulan:</b> {fmt_num(total_pred)} orang",
                "success"
            )

    # ── Footer ───────────────────────────────────────────────────────────────
    divider()
    app_footer("LalinudSARIMA · Prediksi Penumpang Domestik Bandara YIA.")


