import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ===============================
# App Configuration
# ===============================
st.set_page_config(
    page_title="Bike Rental Prediction",
    layout="centered"
)

st.title("🚲 Bike Rental Demand Prediction")
st.write("Predict the number of bike rentals based on weather and seasonal conditions.")

# ===============================
# Load Trained Model
# ===============================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ===============================
# User Inputs
# ===============================
season = st.selectbox(
    "Season",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Spring",
        2: "Summer",
        3: "Fall",
        4: "Winter"
    }[x]
)

yr = st.selectbox("Year", [0, 1])  # 0 = 2011, 1 = 2012
mnth = st.slider("Month", 1, 12, 6)
holiday = st.selectbox("Holiday", [0, 1])
weekday = st.slider("Weekday (0=Sun, 6=Sat)", 0, 6, 3)
workingday = st.selectbox("Working Day", [0, 1])

weathersit = st.selectbox(
    "Weather Situation",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Clear",
        2: "Mist / Cloudy",
        3: "Light Snow / Rain",
        4: "Heavy Rain / Snow"
    }[x]
)

temp = st.slider("Temperature (Normalized)", 0.0, 1.0, 0.5)
atemp = st.slider("Feels Like Temperature (Normalized)", 0.0, 1.0, 0.5)
hum = st.slider("Humidity (Normalized)", 0.0, 1.0, 0.5)
windspeed = st.slider("Windspeed (Normalized)", 0.0, 1.0, 0.3)

# ===============================
# Prediction
# ===============================
if st.button("Predict Bike Rentals"):
    input_data = pd.DataFrame([{
        "season": season,
        "yr": yr,
        "mnth": mnth,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"🚴 Estimated Bike Rentals: **{int(prediction)}**")
