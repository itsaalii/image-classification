"""
================================================================================
KLASIFIKASI AI — ANALISIS & PREDIKSI KATEGORI INSTAN (STREAMLIT RUNNER)
================================================================================
"""

import os
import streamlit as st
import streamlit.components.v1 as components

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Klasifikasi Penataan Sepatu AI — Analisis Rapi vs Berantakan",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Sembunyikan default Streamlit Header & Toolbar agar tampilan 100% bersih
st.markdown("""
<style>
    #MainMenu, header, footer {
        visibility: hidden !important;
        height: 0px !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .stApp {
        background: transparent !important;
    }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        margin: 0 !important;
    }
    iframe {
        border: none !important;
        width: 100% !important;
        height: 100vh !important;
        min-height: 980px !important;
    }
</style>
""", unsafe_allow_html=True)

# Baca dan tampilkan file HTML yang persis diberikan oleh pengguna
html_file_path = os.path.join(os.path.dirname(__file__), 'klasifikasi_ai.html')

if os.path.exists(html_file_path):
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=1050, scrolling=True)
else:
    st.error(f"File '{html_file_path}' tidak ditemukan!")
