import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load model dan scaler
with open('model_rf.pkl', 'rb') as f:
    model_rf = pickle.load(f)

with open('model_dt.pkl', 'rb') as f:
    model_dt = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Dashboard Prediksi Harga Mobil Bekas", layout="wide")
st.title("🚗 Prediksi Harga Mobil Bekas")

with st.sidebar:
    st.markdown("**Nama Anggota Kelompok:**")
    st.text("Rauf Hafizh Asmenta - 202210370311209")
    st.text("Arya Mandala Putra - 202210370311183")
    st.text("Kirouch Alqornie Gymnastiar - 2022210370311189")

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

menu = st.sidebar.radio("Navigasi", ["Eksplorasi Data", "Evaluasi Model", "Feature Importance", "Prediksi Harga"])

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

elif menu == "Evaluasi Model":
    st.header("📈 Evaluasi Model")

    # Load hasil prediksi dari file CSV
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

    st.subheader("Visualisasi Aktual vs Prediksi (Custom)")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=y_test, y=y_pred_rf, ax=axs[0], alpha=0.5, color='dodgerblue')
    axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    axs[0].set_title("Random Forest")
    axs[0].set_xlabel("Aktual")
    axs[0].set_ylabel("Prediksi")

    sns.scatterplot(x=y_test, y=y_pred_dt, ax=axs[1], alpha=0.5, color='darkorange')
    axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    axs[1].set_title("Decision Tree")
    axs[1].set_xlabel("Aktual")
    axs[1].set_ylabel("Prediksi")

    st.pyplot(fig)

    st.subheader("Distribusi Residual (Custom)")
    residual_rf = y_test - y_pred_rf
    residual_dt = y_test - y_pred_dt

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(residual_rf, bins=50, kde=True, ax=axs[0], color='dodgerblue')
    axs[0].set_title("Residual - Random Forest")

    sns.histplot(residual_dt, bins=50, kde=True, ax=axs[1], color='darkorange')
    axs[1].set_title("Residual - Decision Tree")

    st.pyplot(fig)

elif menu == "Feature Importance":
    st.header("⭐ Feature Importance (Random Forest)")
    feature_imp = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': model_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
    st.pyplot(fig)

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

    elif option == "Upload Dataset CSV":
        uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)

            df_upload_encoded = pd.get_dummies(df_upload)
            missing_cols = set(X_encoded.columns) - set(df_upload_encoded.columns)
            for col in missing_cols:
                df_upload_encoded[col] = 0
            df_upload_encoded = df_upload_encoded[X_encoded.columns]

            df_upload_scaled = scaler.transform(df_upload_encoded)
            preds = model_rf.predict(df_upload_scaled)

            result = df_upload.copy()
            result['Prediksi_Harga'] = preds
            st.dataframe(result)

            csv = result.to_csv(index=False).encode()
            st.download_button("📥 Download Hasil Prediksi", data=csv, file_name="hasil_prediksi.csv")
