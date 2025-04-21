# app.py - Streamlit Frontend

import streamlit as st
import joblib

# Load model and scaler from disk
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Diabetes Risk Assessment')

# Input form for user data
with st.form("prediction_form"):
    pregnancies = st.number_input('Pregnancies', 0, 20)
    glucose = st.number_input('Glucose Level', 50, 300)
    bp = st.number_input('Blood Pressure (mmHg)', 20, 150)
    skin_thickness = st.number_input('Skin Thickness (mm)', 0, 100)
    insulin = st.number_input('Insulin Level', 0, 900)
    bmi = st.number_input('BMI', 10.0, 50.0)
    dpf = st.number_input('Diabetes Pedigree Function', 0.0, 2.5)
    age = st.number_input('Age', 20, 100)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare feature vector for prediction
    glucose_bmi = glucose * bmi  # Calculate the new feature
    features = [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age, glucose_bmi]]
    
    # Scale features using the saved scaler
    scaled_features = scaler.transform(features)
    
    # Make prediction using the loaded model
    prediction = model.predict(scaled_features)
    
    # Display results to the user
    if prediction[0] == 1:
        st.error("High diabetes risk detected.")
        st.write("Please consult a healthcare professional.")
    else:
        st.success("Low diabetes risk detected.")
