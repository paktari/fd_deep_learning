import streamlit as st
import numpy as np
import pandas as pd
#from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from jcopml.plot import plot_correlation_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,roc_auc_score,confusion_matrix,precision_score
#import torch
#from torch.utils.data import Dataset, DataLoader

st.title('🤖 Data - Preparation')

st.info('Persiapan data')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_excel('data/data_fd.xlsx')
  df

  st.write('**X**')
  X_raw = df.drop('FD', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.FD
  y_raw

with st.expander('Visualisasi Data'):
  st.scatter_chart(data=df, x='kode_umkm', y='FD', color='#ffaa0088')

# Input features

with st.expander('Input features'):
  #st.write('**Input data**')
  #st.write('**Penggabungan Data**')

  # Data preparation
  # Encode X
  df.drop(['kesimpulan'], axis=1, inplace=True)
  columns_to_exclude = ['FD', 'kode_umkm']
  normalized_data = df.copy()
  scaler = StandardScaler()
  columns_to_normalize = [col for col in df.columns if col not in columns_to_exclude]
  normalized_data[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
  data = pd.DataFrame(normalized_data)

  X = data.drop(['FD', 'kode_umkm', 'tahun'], axis=1)
  y = data['FD']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

  from copy import deepcopy
  data_2 = deepcopy(data)
  fig = plt.figure(figsize=(15,10))
  Numerical=data_2.select_dtypes(exclude='object').columns.tolist()
  st.write(Numerical)
  Numerical = df[Numerical]
  st.write(Numerical.describe())
  Numerical.hist(bins=50)
  st.pyplot(fig)

  
#  st.bar_chart(Numerical, x="tahun", y="x1", color="#ffaa0088")
#  st.bar_chart(Numerical, x="tahun", y="x2", color="#ffaa0099")
#  st.bar_chart(Numerical, x="tahun", y="x3", color="#ffaa0100")

  st.write('UMKM kategori non-financial distress:', round(df["FD"].value_counts()[0]/len(df) * 100,2), '% of data set')
  st.write('UMKM kategori financial distress:', round(df["FD"].value_counts()[1]/len(df) * 100,2), '% of data set')
  st.write('UMKM kategori berpotensi financial distress:', round(df["FD"].value_counts()[2]/len(df) * 100,2), '% of data set')


  fig = plt.figure(figsize = (6,4))
  sns.countplot(x=df["FD"])
  plt.title("Target Distribusi", fontsize=14)
  st.pyplot(fig)
  
  st.write("Korelasi Data")
  kol = [ 'x1', 'x2', 'x3', 'x4', 'x5', 'z', 'FD']
  df_baru = pd.DataFrame(df[kol])
  corr = pd.DataFrame(df_baru.corr().iloc[:13,-1])
  corr
  
  cor_mat=df_baru.corr()
  fig,ax=plt.subplots(figsize=(15,10))
  sns.heatmap(cor_mat,annot=True,linewidths=0.5,fmt=".3f")
  st.write("Grafik Headmap Data")
  st.pyplot(fig)

  st.write("Grafik Korelasi")
  fig= plot_correlation_matrix(data, 'FD', numeric_col='auto')
  st.pyplot(fig)

with st.expander('Data Smothing'):
  
  from imblearn.over_sampling import SMOTE
  import numpy as np
  X = X.replace([np.nan, -np.inf], 0)

  st.write('**Encoded X (input data)**')
  data
  st.write('**Encoded y**')
  y
  smote = SMOTE(random_state=42)
  X_smote_resampled, y_smote_resampled = smote.fit_resample(X, y)

  plt.figure(figsize=(12, 4))

  new_df1 = pd.DataFrame(data=y)

  fig1 = plt.subplot(1, 2, 1)
  new_df1.value_counts().plot(kind='bar',figsize=(10,6),color=['green','blue','red','yellow'])
  plt.title("target sebelum  over sampling dengan SMOTE ")
  plt.xticks(rotation=0);
  st.bar_chart(new_df1)

  fig2 = plt.subplot(1, 2, 2)
  new_df2 = pd.DataFrame(data=y_smote_resampled)

  new_df2.value_counts().plot(kind='bar',figsize=(10,6),color=['green','blue','red','yellow'])
  plt.title("target setelah over sampling dengan SMOTE")
  plt.xticks(rotation=0);

  plt.tight_layout()
  
  st.bar_chart(new_df2)