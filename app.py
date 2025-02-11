import streamlit as st
import requests

# URL of the FastAPI backend
API_URL = "http://127.0.0.1:8000/predict"  # Replace with the actual FastAPI URL if deployed

# Streamlit interface
st.title("Spaceship Titanic Prediction")

# User inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)
cryo_sleep = st.selectbox("CryoSleep", ["True", "False"])
vip = st.selectbox("VIP", ["True", "False"])
cabin = st.text_input("Cabin")
home_planet = st.selectbox("HomePlanet", ["Earth", "Mars", "Unknown"])
destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])

# Prepare the input data for the API
input_data = {
    "Age": age,
    "CryoSleep": cryo_sleep,
    "VIP": vip,
    "Cabin": cabin,
    "HomePlanet": home_planet,
    "Destination": destination
}

# Button to trigger prediction
if st.button("Make Prediction"):
    response = requests.post(API_URL, json=input_data)

    if response.status_code == 200:
        prediction = response.json().get("prediction")
        st.write(f"Prediction: {'Transported' if prediction == 1 else 'Not Transported'}")
    else:
        st.write("Error: Unable to get prediction.")
