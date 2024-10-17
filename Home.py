import streamlit as st
from utilities import load_css

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

from PIL import Image
image = Image.open('company.jpg')

st.image(image, caption='~')


st.header("Selamat Datang! 👋")
st.write('Financial Distress dengan Deep Learning')
st.write(
    """
    # Definisi

    Prediksi kebangkrutan perusahaan adalah proses menggunakan berbagai metode analisis
    untuk mengevaluasi kesehatan keuangan suatu perusahaan dan memprediksi apakah
    perusahaan tersebut berisiko menghadapi kebangkrutan di masa depan.

    Tujuan dari prediksi kebangkrutan perusahaan adalah memberikan informasi kepada
    pemangku kepentingan, seperti pemilik saham, kreditor, dan pemasok, agar mereka dapat
    mengambil langkah-langkah yang tepat untuk mengurangi risiko atau melindungi
    kepentingan mereka.

    Variabel yang akan kami gunakan pada penelitian ini adalah
    1. Penjualan
    2. Laba Kotor
    3. Laba Bersih
    4. Aset Lancar
    5. Aset Tetap
    6. Hutang Jangka Pendek
    7. Hutang Jangka Panjang
    8. Modal Sendiri

    Metode Deep Learning digunakan untuk memprediksi kebangkrutan UKM Batik
    """
)

load_css()
