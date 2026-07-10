# -*- coding: utf-8 -*-
"""
app.py — Main entry point for LalinudSARIMA YIA Dashboard.

Responsibilities:
  - Page config (must be first Streamlit call)
  - CSS injection
  - Sidebar: data upload, info, navigation
  - Routing to page modules
"""
import os
import glob
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from config import BASE_DIR, PAGES
from styles import inject_css
from data_utils import load_and_preprocess

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LalinudSARIMA YIA · Prediksi Penumpang",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy page imports (avoids loading unused modules) ──────────────────────────
def _load_page(name: str):
    if name == "dashboard":
        from pages.dashboard    import render
    elif name == "model":
        from pages.model        import render
    elif name == "download":
        from pages.download     import render
    else:
        def render(): st.error(f"Halaman '{name}' tidak ditemukan.")
    return render



# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        # Brand block
        st.markdown("""
<div class="sb-brand">
  <div class="sb-brand-eyebrow">Sistem Prediksi</div>
  <h2>LalinudSARIMA YIA</h2>
  <p>Penumpang Domestik<br>Metode SARIMA · s = 12</p>
</div>""", unsafe_allow_html=True)

        # Data source
        st.markdown("### 📂 Sumber Data")
        uploaded = st.file_uploader(
            "Upload CSV (opsional)", type=["csv"],
            help="Kosongkan untuk menggunakan data default (data_penumpang_domestik_yia.csv)",
        )

        # Resolve file bytes
        default_csv = os.path.join(BASE_DIR, "data_penumpang_domestik_yia.csv")
        if not os.path.exists(default_csv):
            candidates = (
                glob.glob(os.path.join(BASE_DIR, "*domestik*yia*.csv")) +
                glob.glob(os.path.join(BASE_DIR, "*.csv"))
            )
            if candidates:
                default_csv = candidates[0]

        if uploaded:
            file_bytes = uploaded.read()
        else:
            with open(default_csv, "rb") as f:
                file_bytes = f.read()

        # Load / cache data
        if (
            "file_bytes" not in st.session_state
            or st.session_state["file_bytes"] != file_bytes
        ):
            st.session_state["file_bytes"]   = file_bytes
            st.session_state["model_result"] = None
            with st.spinner("Memuat data…"):
                st.session_state["data"] = load_and_preprocess(file_bytes)

        # Data info chip
        if "data" in st.session_state:
            _, _, ts = st.session_state["data"]
            st.markdown(f"""
<div class="box box-info" style="font-size:12px;margin-top:8px;">
  📅 <b>{ts.index[0].strftime('%b %Y')}</b> —
     <b>{ts.index[-1].strftime('%b %Y')}</b><br>
  📊 <b>{len(ts)}</b> observasi bulanan
</div>""", unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Navigation
        st.markdown("### 🗺️ Navigasi")
        selected_label = st.radio(
            "Pilih Halaman", list(PAGES.keys()), label_visibility="collapsed"
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Model status chip
        m = st.session_state.get("model_result")
        if m:
            st.markdown(f"""
<div class="box box-success" style="font-size:12px;">
  ✅ <b>Model Aktif</b><br>
  SARIMA{m['order']}×{m['seasonal_order'][:3]}<br>
  MAPE: {m['mape']:.2f}% · Akurasi: {m['acc']:.2f}%
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="box box-warn" style="font-size:12px;">
  ⚠️ Model belum dijalankan.<br>
  Buka <b>🤖 Model & Prediksi</b> untuk mulai.
</div>""", unsafe_allow_html=True)

    return PAGES[selected_label]


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()

    if "model_result" not in st.session_state:
        st.session_state["model_result"] = None

    page_key = render_sidebar()

    if "data" not in st.session_state:
        st.warning("⚠️ Data tidak tersedia. Pastikan file CSV ada di direktori yang sama dengan app.py.")
        return

    render_fn = _load_page(page_key)
    render_fn()


if __name__ == "__main__":
    main()
