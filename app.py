import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🩺 AI Symptom Checker")

# User input
user_input = st.text_input("Enter your symptoms (e.g., 'runny nose, headache, body pain')")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some symptoms.")
    else:
        # Transform input using vectorizer
        user_input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(user_input_vectorized)[0]

        st.success(f"🤖 Our AI thinks your symptoms may indicate: **{prediction}**")
