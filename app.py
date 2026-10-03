import streamlit as st
import numpy as np
import joblib
import logging
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Healt Risk Prediction", layout="wide")


logging.basicConfig(filename='app_system.log', level=logging.INFO)

@st.cache_resource
def load_artifacts():
    model = load_model("model_risiko_kesehatan.keras")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_artifacts()

st.sidebar.header("Input Data Pasien")
with st.sidebar.form("input_form"):
    umur = st.number_input("Umur", min_value=1, max_value=100, value=30)
    tekanan_darah = st.number_input("Teknan Dara(mmHg)", min_value=50, max_value=250, value=120)
    kolesterol = st.number_input("Kolesterol (mg/dL)", min_value=50, max_value=400, value=200)
    bmi = st.number_input("BMI", min_value=10, max_value=60, value=30)
    gula_darah = st.number_input("Gula Darah (mg/dL)", min_value=40, max_value=400, value=100)
    aktivitas_fisik = st.number_input("Aktivitas Fisik(jam/minggu)", min_value=0, max_value=168, value=5)
    perokok = st.selectbox("Riwayat Penyakit Jantung ?", options=['Tidak','Ya'])
    riwayat_keluarga = st.selectbox("Riwayat Penyakit Jantung", options=['Tidak','Ya'])

    submit = st.form_submit_button("Analisis Resiko")

st.title("Dashboard Prediksi Resiko Kesehatan")
st.markdown("------")
if submit:
    p = 1 if perokok == 'Ya' else 0
    rk = 1 if riwayat_keluarga == 'Ya' else 0

    try : 
        feature = np.array([[umur, bmi, tekanan_darah, gula_darah, kolesterol, aktivitas_fisik, p, rk]])
        feature_scaled = scaler.transform(feature)
        prob = model.predict(feature_scaled)
        idx = np.argmax(prob, axis=1)
        label = label_encoder.inverse_transform(idx)
        confidence = np.max(prob)

        col1, col2 = st.columns([1,2])
        with col1 :
            st.subheader("Hasil Prediksi")
            st.metric(label="Tingkat Resiko", value=label[0])
            st.metric(label="Tingkat Kepercayaan", value=f"{confidence * 100:.2f}%")

            if label[0] == "Tinggi":
                st.error("Saran : Segera konsultasikan dengan dokter!")
            elif label[0] == "Sedang":
                st.warning("Saran : Perbaiki pola makan dan tingkatkan aktivitas fisik.")
            else:
                st.success("Saran : Pertahankan gaya hidup sehat anda.")
        with col2:
            st.subheader("Distribusi Probabilitas")
            df_prob = pd.DataFrame({'Risiko':label_encoder.classes_, 'Prob':prob[0]})
            fig = px.bar(df_prob, x='Risiko', y="Prob", color='Risiko', 
                         color_discrete_map={'Rendah':"#00C531", 'Sedang':'#f1c40f', 'Tinggi':'#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e :
        st.error(f"Terjadi kesalahan : {e}")
