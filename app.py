
import streamlit as st
import pandas as pd
import joblib

# Load model
pipeline = joblib.load("bike_pipeline.pkl")

st.set_page_config(page_title="Bike Rental Prediction", page_icon="🚲", layout="centered")

st.title("🚲 Bike Rental Demand Prediction")
st.markdown("Predict total bike rentals using time and weather conditions")

st.divider()

# -------- Layout --------
col1, col2 = st.columns(2)

with col1:
    yr = st.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
    mnth = st.slider("Month", 1, 12)
    hr = st.slider("Hour", 0, 23)
    season = st.selectbox("Season", ["springer", "summer", "fall", "winter"])
    weekday = st.selectbox("Weekday (0=Sun)", list(range(7)))

with col2:
    holiday = st.selectbox("Holiday", ["No", "Yes"])
    workingday = st.selectbox("Working Day", ["No work", "Working Day"])
    weathersit = st.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
    temp = st.slider("Temperature", 0.0, 1.0)
    hum = st.slider("Humidity", 0.0, 1.0)
    windspeed = st.slider("Wind Speed", 0.0, 1.0)

# Derived features
is_peak_hour = 1 if (7 <= hr <= 9 or 17 <= hr <= 19) else 0
is_weekend = 1 if weekday in [0, 6] else 0
weather_comfort = temp * (1 - hum)

# Input dataframe
input_df = pd.DataFrame([{
    "yr": yr,
    "mnth": mnth,
    "hr": hr,
    "temp": temp,
    "atemp": temp,   # simple approx
    "hum": hum,
    "windspeed": windspeed,
    "weather_comfort": weather_comfort,
    "season": season,
    "weekday": weekday,
    "holiday": holiday,
    "workingday": workingday,
    "weathersit": weathersit,
    "is_peak_hour": is_peak_hour,
    "is_weekend": is_weekend
}])

st.divider()

if st.button(" Predict Rentals"):
    prediction = pipeline.predict(input_df)[0]
    st.success(f"Estimated Bike Rentals: **{int(prediction)}** ")

st.caption("Model: Gradient Boosting Regressor ")
