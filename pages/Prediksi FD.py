import numpy as np
import pickle
import streamlit as st
from datetime import date

# loaded_model 
loaded_model = pickle.load(open('model/trained_model.pkl', 'rb'))


def main():
    # judul
    st.markdown("<h1 style='text-align: center; color: red;'>Prediksi Financial Distress UMKM</h1>",
                unsafe_allow_html=True)
    st.write('\n')
    # Menu
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


    # input data user

    nama = st.text_input('Nama UMKM')
    penjualan = st.number_input('penjualan')
    lb_kotor = st.number_input("Laba Kotor")
    lb_bersih = st.number_input("Laba Bersih")
    as_lancar = st.number_input("Aset Lancar")
    as_tetap = st.number_input("Aset Tetap")
    ht_pendek = st.number_input("Hutang Jangka Pendek")
    ht_panjang = st.number_input("Hutang Jangka Panjang")
    md_sendiri = st.number_input("Modal Sendiri")
    

    # Hitung Prediksi

    if st.button('Hasil Prediksi Financial Distress'):   
        a = 0.12 * (as_lancar/ht_pendek)
        b = 0.14 * (lb_kotor/penjualan)
        c = 0.33 * (lb_bersih/(ht_pendek+ht_panjang))
        d = 0.06 * (lb_bersih/(ht_pendek+ht_panjang))
        e = 0.99 * (penjualan / (as_lancar+as_tetap))
        fd = a + b + c + d + e
   
        if (fd >= 2.99):
            st.error(nama +" ("+str(fd)+") " + ' tidak dalam kondisi Financial Distress')
        else:
            if (fd > 1.81 ):
                st.error(nama +" ("+str(fd)+") " ' dalam kondisi Abu-abu menuju Financial Distress')
            else:
                st.error(nama +" ("+str(fd)+") " ' Kondisi Financial Distress')

    st.write('\n\n')
    st.write("\n© 2024 Prediksi Financial Distress UMKM Batik.")


if __name__ == '__main__':
    main()
