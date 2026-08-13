import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Title & Layout
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Prediction App")
st.write("Enter patient biological parameters below to calculate diabetes risk.")

# Load Trained Model and Scaler
@st.cache_resource
def load_assets():
    model = joblib.load('models/random_forest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("Model/Scaler files not found. Please ensure 'models/random_forest.pkl' exists.")

# Input Form
st.header("Patient Data Input")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)

# Prediction Logic
if st.button("Predict Risk", type="primary"):
    user_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                             columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Impute zero values if any physiological impossibilities
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        if user_data[col].iloc[0] == 0:
            user_data[col] = np.nan
    
    prediction = model.predict(user_data)[0]
    prediction_proba = model.predict_proba(user_data)[0][1]
    
    st.markdown("---")
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes** (Probability: {prediction_proba * 100:.1f}%)")
    else:
        st.success(f"✅ **Low Risk of Diabetes** (Probability: {(1 - prediction_proba) * 100:.1f}%)")
