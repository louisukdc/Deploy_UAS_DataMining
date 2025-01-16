import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the saved model
try:
    loaded_model = pickle.load(open('/workspaces/Deploy_UAS_DataMining/streamlit/finalized_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please make sure the file is in the correct location and try again.")
    st.stop() # Stop execution if model is not found

# Title of the app
st.title("Diabetes Classification App")

# Sidebar for input features
st.sidebar.header("Input Features")

Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
Glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=100)
BloodPressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
SkinThickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=1000, value=80)
BMI = st.sidebar.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
DiabetesPedigreeFunction = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
Age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)

# Prepare input data for the model
input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

# Make a prediction using the loaded model
prediction = loaded_model.predict(input_data)

# Display the classification result
st.header("Classification Result:")
if prediction[0] == 0:
    st.write("Not Diabetic")
else:
    st.write("Diabetic")

    # Further classification to determine if diabetes is severe (ganas) or moderate (jinak)
    glucose_level = input_data[0][1]  # Glucose level from input_data
    bmi = input_data[0][5]  # BMI from input_data

    # Example of a rule-based check (can be customized as needed)
    if glucose_level > 200 or bmi > 35:  # Example threshold for severe diabetes
        st.warning("Warning: Diabetes may be severe (ganas)")
    else:
        st.write("Diabetes is controlled or moderate (jinak)")

