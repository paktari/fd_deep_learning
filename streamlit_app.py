import streamlit as st
from utilities import load_css

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

st.header("Selamat Datang! 👋")
st.write('Financial Distress deep learning')

load_css()
