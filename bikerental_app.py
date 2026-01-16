import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Bike Rental Prediction")

st.title("🚲 Bike Rental Demand Prediction")

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ===============================
# User Inputs
# ===============================
season = st.selectbox("Season", [1, 2, 3, 4])
yr = st.selectbox("Year (0=2011, 1=2012)", [0, 1])
mnth = st.slider("Month", 1, 12, 6)
holiday = st.selectbox("Holiday", [0, 1])
weekday = st.slider("Weekday (0=Sun, 6=Sat)", 0, 6, 3)
workingday = st.selectbox("Working Day", [0, 1])
weathersit = st.selectbox("Weather Situation", [1, 2, 3, 4])
temp = st.slider("Temperature (Normalized)", 0.0, 1.0, 0.5)
atemp = st.slider("Feels Like Temperature (Normalized)", 0.0, 1.0, 0.5)
hum = st.slider("Humidity (Normalized)", 0.0, 1.0, 0.5)
windspeed = st.slider("Windspeed (Normalized)", 0.0, 1.0, 0.3)

# ===============================
# Prediction
# ===============================
if st.button("Predict Bike Rentals"):

    input_data = pd.DataFrame([[
        int(season),
        int(yr),
        int(mnth),
        int(holiday),
        int(weekday),
        int(workingday),
        int(weathersit),
        float(temp),
        float(atemp),
        float(hum),
        float(windspeed)
    ]], columns=[
        "season", "yr", "mnth", "holiday", "weekday",
        "workingday", "weathersit",
        "temp", "atemp", "hum", "windspeed"
    ])

    prediction = model.predict(input_data)

    st.success(f"🚴 Estimated Bike Rentals: **{int(prediction[0])}**")
