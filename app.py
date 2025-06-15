import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import gdown
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# SETTING STREAMLIT PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Dashboard Prediksi Harga Mobil Bekas", layout="wide")
st.title("🚗 Prediksi Harga Mobil Bekas")

# ----------------------------
# DOWNLOAD FILE DARI GOOGLE DRIVE
# ----------------------------

# Fungsi download file dari Google Drive
def download_file(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        if not os.path.exists(dest_path):
            gdown.download(url, dest_path, quiet=False)
        else:
            print(f"{dest_path} already exists, skipping download.")
    except Exception as e:
        st.error(f"Gagal download file: {dest_path}. Error: {e}")


# ID file dari Google Drive (yang sudah kamu ambil dari folder Drive kamu)
file_ids = {
    'model_rf': '1GxmlfsYQhXUwWzGi-dUBJ2HqVoi9_S54',
    'model_dt': '17cldqhIUn7xi6nINpR3FsasiH4FmoZFJ',
    'scaler': '1h99m7raPnbzE9IaLtxEDP_PlTv_aupnB',
    'cars': '1Exw33SYvP6oCZ1AgVKDlfjmVRyzupriY',
    'y_test': '1LlPVaGsf4ZGc0ni0lDxAideblh4DMiB-',
    'y_pred_rf': '1uBary2mn4dMqedDMe0GWVlJ1PvfkaeNB',
    'y_pred_dt': '12JP6ETK1h3Bc0xf4jU-Sg_uz1VCQsLEI'
}

# Download semua file
download_file(file_ids['model_rf'], 'model_rf.pkl')
download_file(file_ids['model_dt'], 'model_dt.pkl')
download_file(file_ids['scaler'], 'scaler.pkl')
download_file(file_ids['cars'], 'cars.csv')
download_file(file_ids['y_test'], 'y_test.csv')
download_file(file_ids['y_pred_rf'], 'y_pred_rf.csv')
download_file(file_ids['y_pred_dt'], 'y_pred_dt.csv')

# ----------------------------
# LOAD MODEL DAN DATASET
# ----------------------------
with open('model_rf.pkl', 'rb') as f:
    model_rf = pickle.load(f)
with open('model_dt.pkl', 'rb') as f:
    model_dt = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Nama Anggota Kelompok
with st.sidebar:
    st.markdown("**Nama Anggota Kelompok:**")
    st.text("Rauf Hafizh Asmenta - 202210370311209")
    st.text("Arya Mandala Putra - 202210370311183")
    st.text("Kirouch Alqornie Gymnastiar - 2022210370311189")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv('cars.csv')

df = load_data()

selected_features = [
    'odometer_value', 'year_produced', 'engine_capacity', 'has_warranty',
    'drivetrain', 'transmission', 'engine_fuel',
    'number_of_photos', 'duration_listed', 'up_counter'
]
target = 'price_usd'

df_selected = df[selected_features + [target]].dropna()
X_raw = df_selected[selected_features]
y = df_selected[target]

X_encoded = pd.get_dummies(X_raw)
X_scaled = scaler.transform(X_encoded)

# ----------------------------
# MENU STREAMLIT
# ----------------------------

menu = st.sidebar.radio("Navigasi", ["Eksplorasi Data", "Evaluasi Model", "Feature Importance", "Prediksi Harga"])

# ----------------------------
# EKSPLORASI DATA
# ----------------------------

if menu == "Eksplorasi Data":
    st.header("📊 Eksplorasi Data")

    st.subheader("Heatmap Korelasi")
    corr = df_selected.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Distribusi Harga Mobil (USD)")
    fig, ax = plt.subplots()
    sns.histplot(df_selected['price_usd'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Tahun Produksi vs Harga")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_selected['year_produced'], y=df_selected['price_usd'], ax=ax)
    st.pyplot(fig)

# ----------------------------
# EVALUASI MODEL
# ----------------------------

elif menu == "Evaluasi Model":
    st.header("📈 Evaluasi Model")

    y_test = pd.read_csv("y_test.csv").squeeze()
    y_pred_rf = pd.read_csv("y_pred_rf.csv").squeeze()
    y_pred_dt = pd.read_csv("y_pred_dt.csv").squeeze()

    st.subheader("Perbandingan Metrik Evaluasi")
    eval_df = pd.DataFrame({
        'MAE': [mean_absolute_error(y_test, y_pred_rf), mean_absolute_error(y_test, y_pred_dt)],
        'RMSE': [np.sqrt(mean_squared_error(y_test, y_pred_rf)), np.sqrt(mean_squared_error(y_test, y_pred_dt))],
        'R2': [r2_score(y_test, y_pred_rf), r2_score(y_test, y_pred_dt)]
    }, index=['Random Forest', 'Decision Tree'])
    st.dataframe(eval_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    eval_df[['MAE', 'RMSE', 'R2']].plot(kind='bar', ax=ax, colormap='viridis')
    plt.xticks(rotation=0)
    st.pyplot(fig)

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------

elif menu == "Feature Importance":
    st.header("⭐ Feature Importance (Random Forest)")
    feature_imp = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': model_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
    st.pyplot(fig)

# ----------------------------
# PREDIKSI HARGA MOBIL
# ----------------------------

elif menu == "Prediksi Harga":
    st.header("🔮 Prediksi Harga Mobil")

    option = st.radio("Metode Input", ["Input Manual", "Upload Dataset CSV"])

    if option == "Input Manual":
        st.subheader("Form Input Data Mobil")

        odometer_value = st.number_input('Odometer Value (km)', 0, 1000000, 50000)
        year_produced = st.number_input('Tahun Produksi', 1950, 2025, 2015)
        engine_capacity = st.number_input('Engine Capacity (L)', 0.0, 10.0, 2.0, step=0.1)
        has_warranty = st.selectbox('Garansi', ['Tidak', 'Ya'])
        drivetrain = st.selectbox('Drivetrain', ['front', 'rear', 'all'])
        transmission = st.selectbox('Transmisi', ['mechanical', 'automatic'])
        engine_fuel = st.selectbox('Jenis Bahan Bakar', ['gasoline', 'diesel', 'gas', 'hybrid', 'electric'])
        number_of_photos = st.number_input('Jumlah Foto Iklan', 0, 50, 5)
        duration_listed = st.number_input('Durasi Iklan (hari)', 0, 1000, 30)
        up_counter = st.number_input('Frekuensi Update Iklan', 0, 100, 5)

        data_input = pd.DataFrame({
            'odometer_value': [odometer_value],
            'year_produced': [year_produced],
            'engine_capacity': [engine_capacity],
            'has_warranty': [1 if has_warranty == 'Ya' else 0],
            'drivetrain': [drivetrain],
            'transmission': [transmission],
            'engine_fuel': [engine_fuel],
            'number_of_photos': [number_of_photos],
            'duration_listed': [duration_listed],
            'up_counter': [up_counter]
        })

        data_input_encoded = pd.get_dummies(data_input)
        missing_cols = set(X_encoded.columns) - set(data_input_encoded.columns)
        for col in missing_cols:
            data_input_encoded[col] = 0
        data_input_encoded = data_input_encoded[X_encoded.columns]

        data_input_scaled = scaler.transform(data_input_encoded)
        harga_prediksi = model_rf.predict(data_input_scaled)[0]
        st.success(f"Prediksi Harga Mobil: ${harga_prediksi:,.2f}")
