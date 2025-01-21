import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Memuat model yang telah disimpan
try:
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan file berada di lokasi yang benar dan coba lagi.")
    st.stop()  # Menghentikan eksekusi jika model tidak ditemukan

# Judul aplikasi
st.title("Aplikasi Klasifikasi Diabetes")

# Sidebar untuk input fitur
st.sidebar.header("Masukkan Fitur")

Pregnancies = st.sidebar.number_input("Kehamilan", min_value=0, max_value=20, value=0)
Glucose = st.sidebar.number_input("Glukosa", min_value=0, max_value=200, value=100)
BloodPressure = st.sidebar.number_input("Tekanan Darah", min_value=0, max_value=150, value=70)
SkinThickness = st.sidebar.number_input("Ketebalan Kulit", min_value=0, max_value=100, value=20)
Insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=1000, value=80)
BMI = st.sidebar.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, max_value=60.0, value=25.0)
DiabetesPedigreeFunction = st.sidebar.number_input("Fungsi Riwayat Keluarga Diabetes", min_value=0.0, max_value=3.0, value=0.5)
Age = st.sidebar.number_input("Usia", min_value=0, max_value=100, value=30)

# Menyiapkan data input untuk model
input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

# Membuat prediksi menggunakan model yang telah dimuat
prediction = loaded_model.predict(input_data)

# Menampilkan hasil klasifikasi
st.header("Hasil Klasifikasi:")
if prediction[0] == 0:
    st.write("Tidak Mengidap Diabetes")
else:
    # Klasifikasi menjadi DM1 atau DM2
    glucose_level = input_data[0][1]  # Tingkat glukosa dari input_data
    age = input_data[0][7]  # Usia dari input_data
    
    if age < 30 and glucose_level > 150:  # Aturan sederhana untuk Diabetes Melitus Tipe 1
        st.warning("Hasil: Diabetes Melitus Tipe 1 (DM1)")
        st.info("""
### Edukasi tentang Diabetes Melitus Tipe 1 (DM1):
1. Diabetes Melitus Tipe 1 biasanya terjadi pada usia muda, terutama anak-anak dan remaja.
2. Disebabkan oleh kerusakan pada sel beta pankreas sehingga tubuh tidak dapat memproduksi insulin.
3. Gejala meliputi sering haus, sering buang air kecil, penurunan berat badan mendadak, dan kelelahan ekstrem.
4. Penanganan: memerlukan terapi insulin seumur hidup dan pola makan sehat.
        """)
    else:  # Asumsikan sisanya sebagai Diabetes Melitus Tipe 2
        st.warning("Hasil: Diabetes Melitus Tipe 2 (DM2)")
        st.info("""
### Edukasi tentang Diabetes Melitus Tipe 2 (DM2):
1. Diabetes Melitus Tipe 2 lebih sering terjadi pada usia dewasa atau lanjut usia, tetapi kini juga ditemukan pada usia muda.
2. Disebabkan oleh resistensi insulin (tubuh tidak dapat menggunakan insulin secara efektif) atau produksi insulin yang tidak mencukupi.
3. Faktor risiko: kelebihan berat badan, kurang aktivitas fisik, dan riwayat keluarga dengan diabetes.
4. Penanganan: mencakup perubahan gaya hidup, obat-obatan, dan kadang terapi insulin.
        """)

