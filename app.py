import streamlit as st
import numpy as np
import pickle

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ model.pkl created successfully")

st.set_page_config(page_title="Bike Rental Prediction", layout="centered")

st.title("🚲 Bike Rental Prediction")
st.write("Predict daily bike rentals using environmental and seasonal data")

# User inputs
season = st.selectbox("Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)", [1, 2, 3, 4])
yr = st.selectbox("Year (0=2011, 1=2012)", [0, 1])
mnth = st.slider("Month", 1, 12)
holiday = st.selectbox("Holiday", [0, 1])
weekday = st.slider("Weekday (0=Sunday)", 0, 6)
workingday = st.selectbox("Working Day", [0, 1])
weathersit = st.selectbox("Weather Situation (1–4)", [1, 2, 3, 4])
temp = st.slider("Temperature (Normalized)", 0.0, 1.0)
atemp = st.slider("Feels Like Temp (Normalized)", 0.0, 1.0)
hum = st.slider("Humidity (Normalized)", 0.0, 1.0)
windspeed = st.slider("Wind Speed (Normalized)", 0.0, 1.0)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[season, yr, mnth, holiday, weekday,
                             workingday, weathersit, temp,
                             atemp, hum, windspeed]])

    prediction = model.predict(input_data)

    st.success(f"🚴 Estimated Bike Rentals: {int(prediction[0])}")
