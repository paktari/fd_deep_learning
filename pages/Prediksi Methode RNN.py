import streamlit as st
from utilities import load_css

st.set_page_config(page_title="RNN", page_icon="4")

st.header("Prediksi FD UMKM Batik metode RNN")

st.markdown('''Recurrent Neural Network''')

# Recurrent Neural Network

# 1 - Data Preprocessing

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import training set
dataset_train = pd.read_excel('data/data_fd.xlsx')
dataset_train.drop(['kode_umkm','kesimpulan'], axis=1, inplace=True)
training_set = dataset_train.iloc[:, 1:2].values
st.write(dataset_train)
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating data structure
X_train = []
y_train = []
for i in range(60, 246):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 2 - Building the RNN

# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising RNN
regressor = Sequential()

# Tambah LSTM layer dan beberapa Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Tambah LSTM layer kedua
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Tambah LSTM layer ketiga
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Tambah LSTM layer Keempat
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Tambah output layer
regressor.add(Dense(units = 1))

# Compiling RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting RNN dengan Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# 3 - Buat prediksi dan visualisasi hasil

# data real
dataset_test = pd.read_excel('data/data_fd_test.xlsx')
dataset_test.drop(['kode_umkm','kesimpulan'], axis=1, inplace=True)
real_stock_price = dataset_test.iloc[:, 1:2].values
#real_stock_price = dataset_test.FD

# Prediksi FD
dataset_total = pd.concat((dataset_train['FD'], dataset_test['FD']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
st.write('Hasil Prediksi')
#st.write(predicted_stock_price)
#st.write(real_stock_price)

# Visualisasi hasil
fig = plt.figure(figsize=(6,6))
plt.plot(real_stock_price, color = 'red', label = 'Real FD UMKM')
plt.plot(predicted_stock_price, color = 'blue', label = 'Prediksi FD UMKM')
plt.title('Prediksi FD UMKM Batik')
plt.xlabel('Tahun')
plt.ylabel('FD')
plt.legend()
st.pyplot(fig)