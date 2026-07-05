# -*- coding: utf-8 -*-
"""pages/diagnostics.py — Residual diagnostic tests and charts."""
import streamlit as st
from components import section_header, page_hero, info_box
from chart_utils import chart_residual_diagnostics


def render() -> None:
    m = st.session_state.get("model_result")

    page_hero(
        "Validasi Model",
        "Diagnostik Residual",
        "Uji Ljung-Box (autokorelasi) · Shapiro-Wilk (normalitas) · Grafik diagnostik.",
    )

    if not m:
        info_box(
            "⚠️ Model belum dijalankan. "
            "Buka halaman <b>🤖 Model & Prediksi</b> terlebih dahulu.",
            "warn",
        )
        return

    residuals = m["residuals"]
    lb_ok   = m["lb_p"]   > 0.05
    norm_ok = m["norm_p"] > 0.05

    # ── Summary metrics ──────────────────────────────────────────────────────
    section_header("Ringkasan", "Hasil Uji Diagnostik")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rata-rata Residual", f"{residuals.mean():.4f}",
              help="Idealnya mendekati 0")
    c2.metric("Std. Residual",       f"{residuals.std():.2f}")
    c3.metric("Ljung-Box p-value",   f"{m['lb_p']:.4f}",
              delta="OK" if lb_ok else "Perlu Cek",
              delta_color="normal" if lb_ok else "inverse")
    c4.metric("Shapiro-Wilk p-value", f"{m['norm_p']:.4f}",
              delta="Normal" if norm_ok else "Tidak Normal",
              delta_color="normal")

    # ── Diagnostic charts ────────────────────────────────────────────────────
    section_header("Visualisasi", "Plot Diagnostik Residual",
                   "Residual vs waktu · Distribusi · Q-Q · ACF")
    st.plotly_chart(chart_residual_diagnostics(residuals), use_container_width=True)

    # ── Interpretation ───────────────────────────────────────────────────────
    section_header("Interpretasi", "Kesimpulan Diagnostik")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""**Ljung-Box Test (Autokorelasi Residual)**
- Statistik: `{m['lb_stat']:.4f}`
- p-value: `{m['lb_p']:.4f}`
- **{'✅ Tidak ada autokorelasi signifikan' if lb_ok else '⚠️ Masih ada autokorelasi'}**

> H₀: tidak ada autokorelasi → tolak H₀ jika p ≤ 0.05""")
    with col_b:
        st.markdown(f"""**Shapiro-Wilk Test (Normalitas Residual)**
- p-value: `{m['norm_p']:.4f}`
- **{'✅ Residual berdistribusi normal' if norm_ok else 'ℹ️ Tidak normal (umum di time series)'}**

> H₀: residual berdistribusi normal → tolak H₀ jika p ≤ 0.05""")

    # ── Overall verdict ──────────────────────────────────────────────────────
    if lb_ok:
        info_box(
            "✅ Residual bersifat <b>white noise</b> — tidak ada pola tersisa "
            "yang belum ditangkap model. Model dianggap memadai.",
            "success",
        )
    else:
        info_box(
            "⚠️ Masih terdapat autokorelasi di residual. "
            "Pertimbangkan untuk menambah orde AR atau MA.",
            "warn",
        )
