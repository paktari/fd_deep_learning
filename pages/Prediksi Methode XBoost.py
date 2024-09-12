import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,roc_auc_score,confusion_matrix,precision_score
import seaborn as sns
#from torch.utils.data import Dataset, DataLoader
#import torch

from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,roc_auc_score,confusion_matrix,precision_score
st.title("Prediksi Financial Distress methode XBoost")
st.markdown("Prediksi Financial Distress data UMKM Batik menggunakan Metode XBoost")

df = pd.read_excel('data/data_fd.xlsx')
st.write("Data Sumber")
st.write(df)
df.drop(['kesimpulan'], axis=1, inplace=True)

def evaluation(Y_test,Y_pred):
    acc = accuracy_score(Y_test,Y_pred)
    rcl = recall_score(Y_test,Y_pred,average = 'weighted')
    f1 = f1_score(Y_test,Y_pred,average = 'weighted')
    ps = precision_score(Y_test,Y_pred,average = 'weighted')

    metric_dict={'accuracy': round(acc,3),
               'recall': round(rcl,3),
               'F1 score': round(f1,3),
               'Precision score': round(ps,3)
              }

    return print(metric_dict)

# membagi data x dan y kemudian melakukan train test split
X = df.drop(['FD', 'kode_umkm', 'tahun'], axis=1)
y = df['FD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# melakukan pelatihan model
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# menampilkan akurasi dan mengebaluasi model xgboost
y_pred_xgb = xgb_model.predict(X_test)
st.write("\nXGBoost Model:")
accuracy_xgb_smote = round(accuracy_score(y_test, y_pred_xgb),3)
st.write("Accuracy:",accuracy_xgb_smote)
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred_xgb))

# menampilkan confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb)
fig = plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()
st.pyplot(fig)

xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the XGBoost model
st.write("\nXGBoost Model:")
accuracy_xgb_smote = round(accuracy_score(y_test, y_pred_xgb),3)
st.write("Accuracy:",accuracy_xgb_smote)
st.write("Hasil Klasifikasi:")
st.write(classification_report(y_test, y_pred_xgb))

st.write(evaluation(y_test,y_pred_xgb))

#st.write(evaluation(y_test,y_pred_xgb))

cm = confusion_matrix(y_test, y_pred_xgb)

fig = plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')
st.pyplot(fig)