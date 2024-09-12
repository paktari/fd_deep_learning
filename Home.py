import streamlit as st
from utilities import load_css

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)
st.markdown("<h1 style='text-align: center; color: red;'>Prediksi Financial Distress UMKM</h1>", unsafe_allow_html=True)
st.write('\n')
# User Information
st.sidebar.markdown("""
        1. Nama UMKM
        2. Penjualan
        3. Laba Kotor
        4. Laba Bersih
        """)
st.header("Selamat Datang! 👋")
st.write('Financial Distress deep learning')

load_css()
