import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model and scaler
model = pickle.load(open("best_model.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))

# Define feature names
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Enter the details below to predict whether a person has diabetes.")

# User Input Form
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=30)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)

# Convert input to NumPy array and reshape
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                        insulin, bmi, diabetes_pedigree, age]])

# Convert to DataFrame with feature names
user_input_df = pd.DataFrame(user_input, columns=feature_names)

# Scale the input data
user_input_scaled = scaler.transform(user_input_df)

# Predict Diabetes
if st.button("Predict"):
    prediction = model.predict(user_input_scaled)
    prediction_proba = model.predict_proba(user_input_scaled)[0][1] * 100  # Probability of having diabetes

    if prediction[0] == 1:
        st.error(f"⚠️ The person **has diabetes** with a confidence of {prediction_proba:.2f}%")
    else:
        st.success(f"✅ The person **does not have diabetes** with a confidence of {100 - prediction_proba:.2f}%")

st.write("This prediction is based on a machine learning model trained on historical diabetes data.")
