import streamlit as st
from utilities import load_css

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)
st.markdown("<h1 style='text-align: center; color: red;'>Prediksi Financial Distress UMKM</h1>",
                unsafe_allow_html=True)
    st.write('\n')
    # User Information
    st.sidebar.markdown("""
    Data yang perlu diisi : 
                
        1. Nama UMKM
        2. Penjualan
        3. Laba Kotor
        4. Laba Bersih
        5. Aset Lancar
        6. Aset Tetap
        7. Hutang Jangka Pendek
        8. Hutang Jangka Panjang
        9. Modal Sendiri
        """)

st.header("Selamat Datang! 👋")
st.write('Financial Distress deep learning')

load_css()
