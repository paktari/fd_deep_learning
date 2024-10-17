import numpy as np
import pickle
import streamlit as st
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

# loaded_model 
loaded_model = pickle.load(open('model/trained_model.pkl', 'rb'))


def main():
    # judul
    st.markdown("<h3 style='text-align: center; color: red;'>Prediksi Tahunan Financial Distress UKM Batik</h3>",
                unsafe_allow_html=True)
    st.write('\n')
   
    data_file = st.file_uploader("Upload Data Financial",type=['xls','xlsx','csv'])

    if data_file is not None:
        df = pd.read_excel(data_file)
        st.write('Data Keuangan UKM')
        st.dataframe(df)
        df['fd']=0
        df['ket']=''
        df['x1']=0
        df['x2']=0
        df['x3']=0
        df['x4']=0
        df['x5']=0

    if st.button('Prediksi Financial Distress'):
 # ambil data 
        i = 0
        while i <= len(df):
            penjualan = df['penjualan']
            lb_kotor = df['lb_kotor']
            lb_bersih = df['lb_bersih']
            as_lancar = df['aset_lancar']
            as_tetap = df['aset_tetap']
            ht_pendek = df['ht_jk_pd']
            ht_panjang = df['ht_jk_pj']
            md_sendiri = df['modal']
            a = 1.2 * (md_sendiri/(as_lancar+as_tetap))
            b = 1.4 * (lb_kotor/(as_lancar+as_tetap))
            c = 3.3 * (lb_bersih/(as_lancar+as_tetap))
            d = 0.6 * (lb_bersih/(ht_pendek + ht_panjang))
            e = 0.99 * (penjualan/(as_lancar+as_tetap))

            fd = a + b + c + d + e
            
            df['fd']=fd
            df['x1']=a
            df['x2']=b
            df['x3']=c
            df['x4']=d
            df['x5']=e
            i +=1

# 
        for index, row in df.iterrows():
            fd = row['fd']
            if  fd >= 2.99:
           # print(" ("+str(fd)+") " + ' tidak dalam kondisi Financial Distress')
                df.at[index, 'ket']='Aman'
            elif fd > 1.81:
           # print(" ("+str(fd)+") " ' dalam kondisi Abu-abu menuju Financial Distress')
                df.at[index, 'ket']='Abu-abu'
            else:
                df.at[index, 'ket']='FD'
           # print(" ("+str(fd)+") " ' Kondisi Financial Distress')
    # Hitung Prediksi
        st.write('Data Hasil Prediksi Tahunan')
        st.dataframe(df)

        st.markdown("<h4 style='text-align: left; color: red;'>Grafik Prediksi Kebrangkutan</h4>",
                unsafe_allow_html=True)
    #    fig = plt.figure(figsize=(15,10))
    #    plt.xlabel("Tahun")
    #    plt.ylabel("Nilai FD")
    #    plt.plot(round(df['tahun']), df['fd'])
    #    st.pyplot(fig)

        st.bar_chart(df, x="tahun", y=["fd","ket"], color=["#ff0088ff","#ffaa0088"], x_label='Tahun', y_label='Nilai Prediksi FD')

        st.write('Keterangan:')
        st.write('FD >= 2.99 ==> Tidak dalam kondisi Financial Distress')
        st.write('FD > 1.81 ==> dalam kondisi Abu-abu menuju Financial Distress')
        st.write('FD < 1.81 ==> Kondisi Financial Distress')

        st.markdown("<h4 style='text-align: left; black: red;'>Grafik Kinerja Keuangan UKM Batik</h4>",
                unsafe_allow_html=True)

        kol = [ 'penjualan', 'lb_kotor', 'lb_bersih','tahun']
        df1 = pd.DataFrame(df[kol])
#       st.bar_chart(df1,x='tahun', y=['penjualan', 'lb_kotor', 'lb_bersih'], x_label='Tahun', y_label='Nilai',stack=None)
        st.bar_chart(df, x="tahun", y=["penjualan", "lb_kotor", "lb_bersih", "total_aset", "ht_jk_pd","ht_jk_pj"],
                     x_label='Tahun', y_label='Nilai', stack=None)
        st.write('Keterangan:')
        st.write('penjualan ==> Penjualan')
        st.write('lb_kotor ==> Laba Kotor')
        st.write('lb_bersih ==> Laba Bersih')
        st.write('hk_jk_pd ==> Hutang Jangka Pendek')
        st.write('hk_jk_pj ==> Hutang Jangka Panjang')
        
        # grafik 
     #   df1 = pd.DataFrame(df[['penjualan','tahun']])
     #   df2 = pd.DataFrame(df[['lb_kotor','tahun']])
     #   df3 = pd.DataFrame(df[['lb_bersih','tahun']])
     #   df4 = pd.DataFrame(df[['ht_jk_pd','tahun']])
     #   cols = st.columns(2)
     #   fig1 = px.bar(df1, x="tahun", y="penjualan",title= "Kinerja Penjualan")
     #   cols[0].plotly_chart(fig1)
     #   fig2 = px.bar(df2, x="tahun", y="lb_kotor",title= "Kinerja Laba Kotor")
     #   cols[1].plotly_chart(fig2)
     #   kols = st.columns(2)
     #   fig3 = px.bar(df3, x="tahun", y="lb_bersih",title= "Kinerja Laba Bersih")
     #   kols[0].plotly_chart(fig3)
     #   fig4 = px.bar(df4, x="tahun", y="ht_jk_pd",title= "Posisi Hutang Jangka Pendek")
     #   kols[1].plotly_chart(fig4)
        # batas
        st.markdown("<h4 style='text-align: left; black: red;'>Grafik Rasio Keuangan UKM Batik</h4>",
                unsafe_allow_html=True)
        
#alternatif grafik
        st.line_chart(df, x="tahun", y=["x1", "x2", "x3", "x4", "x5"])
        st.write('Keterangan:')
        st.write('x1 ==> Rasio Likuiditas')
        st.write('x2 ==> Rasio Profitabilitas')
        st.write('x3 ==> Rasio Aset Produktif')
        st.write('x4 ==> Rasio Solvency')
        st.write('x5 ==> Rasio Efisiensi Operasi')    

    st.write('\n\n')
    st.write("\n© 2024 Prediksi Financial Distress UMKM Batik.")


if __name__ == '__main__':
    main()
