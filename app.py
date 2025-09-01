import streamlit as st
import joblib
import pandas as pd

# Load trained model
try:
    model = joblib.load("model.pkl")
except:
    model = None

st.title("🩺 AI Symptom Checker")
st.write("Enter your symptoms and get a quick prediction.")

# Text input for symptoms
user_input = st.text_input("Describe your symptoms here:")

if st.button("Check"):
    if model is None:
        st.error("Model not found! Please train and save model.pkl first.")
    elif user_input.strip() == "":
        st.warning("Please enter some symptoms first.")
    else:
        # Dummy example: in your real version, convert input -> features
        prediction = model.predict([user_input])  
        st.success(f"Prediction: {prediction[0]}")
    