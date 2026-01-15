import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="wide"
)

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🚲 Bike Rental Demand Prediction")

st.sidebar.header("Input Parameters")

season = st.sidebar.selectbox("Season", [1, 2, 3, 4])
yr = st.sidebar.selectbox("Year (0=2018, 1=2019)", [0, 1])
mnth = st.sidebar.slider("Month", 1, 12)
hr = st.sidebar.slider("Hour", 0, 23)
holiday = st.sidebar.selectbox("Holiday", [0, 1])
weekday = st.sidebar.selectbox("Weekday", list(range(7)))
workingday = st.sidebar.selectbox("Working Day", [0, 1])
weathersit = st.sidebar.selectbox("Weather", [1, 2, 3, 4])

temp = st.sidebar.slider("Temperature", 0.0, 1.0)
atemp = st.sidebar.slider("Feels Like Temp", 0.0, 1.0)
hum = st.sidebar.slider("Humidity", 0.0, 1.0)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0)

input_df = pd.DataFrame([{
    "season": season,
    "yr": yr,
    "mnth": mnth,
    "hr": hr,
    "holiday": holiday,
    "weekday": weekday,
    "workingday": workingday,
    "weathersit": weathersit,
    "temp": temp,
    "atemp": atemp,
    "hum": hum,
    "windspeed": windspeed
}])

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("🚀 Predict Demand"):
    prediction = model.predict(input_df)[0]
    st.success(f"🚴 Predicted Bike Rentals: {int(prediction)}")

    st.subheader("Input Summary")
    st.dataframe(input_df)
