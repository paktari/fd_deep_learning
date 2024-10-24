import streamlit as st
from utilities import load_css

st.set_page_config(page_title="Deskripsi", page_icon="1️⃣")

st.header("Fiancial Distress UKM Batik dengan Deep Learning")

st.markdown('''
Financial distress adalah penurunan kinerja keuangan perusahaan.
Setiap perusahaan pasti memiliki fase naik dan turun performa, termasuk dari segi keuangan. Financial distress adalah istilah untuk menyebut kondisi penurunan performa tersebut. Apabila tidak segera ditangani sumbernya, financial distress akan berujung pada kebangkrutan. 
Financial distress adalah peristiwa penurunan kinerja keuangan perusahaan secara terus menerus dalam jangka waktu tertentu. Bagi perusahaan, financial distress adalah salah satu kondisi penyebab kebangkrutan paling sering. Sebab berbeda dengan penurunan laba biasa, nominal kerugian karena financial distress bisa sangat besar hingga mempengaruhi kelancaran operasional perusahaan.


''')
#st.markdown("<a href='data_fd_tahunan.xlsx' download>")
#st.download_button( 
#    label="Download template",
#    data=csv,
#    file_name='data_fd_tahunan.xlsx',
#    mime='text/csv',)

load_css()
