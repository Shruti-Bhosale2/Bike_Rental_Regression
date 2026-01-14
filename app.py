import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Bike Rental Prediction", layout="centered")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚲 Bike Rental Prediction App")

st.write("Enter the details to predict bike rentals")

# User Inputs
season = st.selectbox("Season", [1, 2, 3, 4])
yr = st.selectbox("Year (0=2018, 1=2019)", [0, 1])
mnth = st.slider("Month", 1, 12)
hr = st.slider("Hour", 0, 23)
holiday = st.selectbox("Holiday", [0, 1])
workingday = st.selectbox("Working Day", [0, 1])
weathersit = st.selectbox("Weather Situation", [1, 2, 3, 4])

temp = st.slider("Temperature", 0.0, 1.0)
atemp = st.slider("Feels Like Temperature", 0.0, 1.0)
hum = st.slider("Humidity", 0.0, 1.0)
windspeed = st.slider("Windspeed", 0.0, 1.0)

# Create DataFrame
input_df = pd.DataFrame([{
    "season": season,
    "yr": yr,
    "mnth": mnth,
    "hr": hr,
    "holiday": holiday,
    "workingday": workingday,
    "weathersit": weathersit,
    "temp": temp,
    "atemp": atemp,
    "hum": hum,
    "windspeed": windspeed
}])

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Rentals: {int(prediction[0])}")

