import streamlit as st
import joblib
import numpy as np 

# Load model dan scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Konsumsi Cannabis")
st.header("Masukkan Data Fitur")
Nscore = st.slider("Neuroticism Score (Nscore)", -3.0, 3.0, 0.0)
Escore = st.slider("Extraversion Score (Escore)", -3.0, 3.0, 0.0)
Oscore = st.slider("Openness Score (Oscore)", -3.0, 3.0, 0.0)
AScore = st.slider("Agreeableness Score (AScore)", -3.0, 3.0, 0.0)
Cscore = st.slider("Conscientiousness Score (Cscore)", -3.0, 3.0, 0.0)
Impulsive = st.slider("Impulsiveness", -3.0, 3.0, 0.0)
SS = st.slider("Sensation Seeking (SS)", -3.0, 3.0, 0.0)

if st.button("Prediksi"):
    input_data = np.array([[Nscore, Escore, Oscore, AScore, Cscore, Impulsive, SS]])
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)[0]

    categories = {
        0: "Tidak pernah",
        1: "Kadang-kadang",
        2: "Sering",
        3: "Sangat sering",
        4: "Hampir selalu",
        5: "Selalu",
    }
    st.write(f"Prediksi tingkat konsumsi cannabis: **{categories[prediction]}**")