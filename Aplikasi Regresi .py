import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from google.colab import files
import io

# Unggah file 'data.csv' ke Google Colab
uploaded = files.upload()
# Membaca data dari file yang diunggah
file_name = next(iter(uploaded))
data = pd.read_csv(io.BytesIO(uploaded[file_name]))
# Menampilkan nama kolom untuk memastikan kesesuaian
print(data.columns)
print(data.head())
# Memilih kolom yang relevan
# Menggunakan kolom sesuai dengan data yang diunggah dari Kaggle
# Kolom 'Sample Question Papers Practiced' untuk NL dan 'Performance Index' untuk NT
NL = data['Sample Question Papers Practiced'].values
NT = data['Performance Index'].values
# Model Linear (Metode 2)
X_NL = NL.reshape(-1, 1)
linear_model_NL = LinearRegression()
linear_model_NL.fit(X_NL, NT)
NT_pred_linear_NL = linear_model_NL.predict(X_NL)
# Plot hasil regresi linear
plt.figure(figsize=(12, 6))
plt.scatter(NL, NT, color='blue', label='Data Asli')
plt.plot(NL, NT_pred_linear_NL, color='red', label='Regresi Linear')
plt.title('Regresi Linear (data Latihan Soal vs Nilai Ujian)')
plt.xlabel('Data Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.grid(True)
plt.show()
# Menghitung RMS untuk model linear (NL)
rms_linear_NL = np.sqrt(mean_squared_error(NT, NT_pred_linear_NL))
print(f'RMS untuk model linear (NL): {rms_linear_NL}')
# Model Eksponensial (Metode 2)
def exp_func(x, a, b, c):
 return a * np.exp(b * x) + c
popt, _ = curve_fit(exp_func, NL, NT)
NT_pred_exp_NL = exp_func(NL, *popt)
# Plot hasil regresi eksponensial
plt.figure(figsize=(12, 6))
plt.scatter(NL, NT, color='green', label='Data Asli')
plt.plot(NL, NT_pred_exp_NL, color='red', label='Regresi Eksponensial')
plt.title('Regresi Eksponensial (Data Latihan Soal vs Nilai Ujian)')
plt.xlabel('Data Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.grid(True)
plt.show()
# Menghitung RMS untuk model eksponensial (NL)
rms_exp_NL = np.sqrt(mean_squared_error(NT, NT_pred_exp_NL))
print(f'RMS untuk model eksponensial (NL): {rms_exp_NL}')