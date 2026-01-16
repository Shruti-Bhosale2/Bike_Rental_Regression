import streamlit as st
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("bike_pipeline.pkl")

# Page settings
st.set_page_config(
    page_title="Bike Rental Prediction",
    page_icon="🚲",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
.big-title {
    font-size:40px;
    font-weight:700;
    color:#1f77b4;
}
.card {
    padding: 25px;
    border-radius: 15px;
    background: #f0f6ff;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
}
.footer {
    text-align:center;
    color: gray;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="big-title"> Bike Rental Demand Prediction</div>', unsafe_allow_html=True)
st.write("Predict bike rental demand using time and weather conditions.")

st.divider()

# Sidebar inputs
st.sidebar.header("⚙️ Input Parameters")

yr = st.sidebar.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
mnth = st.sidebar.slider("Month", 1, 12)
hr = st.sidebar.slider("Hour", 0, 23)

season = st.sidebar.selectbox("Season", ["springer", "summer", "fall", "winter"])
weekday = st.sidebar.selectbox("Weekday (0=Sunday)", list(range(7)))

holiday = st.sidebar.selectbox("Holiday", ["No", "Yes"])
workingday = st.sidebar.selectbox("Working Day", ["No work", "Working Day"])

weathersit = st.sidebar.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])

temp = st.sidebar.slider("Temperature", 0.0, 1.0)
hum = st.sidebar.slider("Humidity", 0.0, 1.0)
windspeed = st.sidebar.slider("Wind Speed", 0.0, 1.0)

# Feature engineering
is_peak_hour = 1 if (7 <= hr <= 9 or 17 <= hr <= 19) else 0
is_weekend = 1 if weekday in [0, 6] else 0
weather_comfort = temp * (1 - hum)

# Input dataframe
input_df = pd.DataFrame([{
    "yr": yr,
    "mnth": mnth,
    "hr": hr,
    "temp": temp,
    "atemp": temp,
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

# Layout
left, right = st.columns([2, 1])

with left:
    st.subheader(" Input Summary")
    st.dataframe(input_df, use_container_width=True)

with right:
    st.subheader(" Prediction")

    if st.button(" Predict Rentals", use_container_width=True):
        prediction = pipeline.predict(input_df)[0]

        # Fix negative values
        prediction = max(0, int(prediction))

        st.markdown(f"""
<div class="card">
    <h3 style="color:#1f4e79;">Estimated Rentals</h3>
    <h1 style="color:#0b3c5d;">{prediction}</h1>
    <p style="color:#4f5d75;">bikes</p>
</div>
""", unsafe_allow_html=True)


st.divider()

# Footer
st.markdown("""
<div class="footer">
Model: Gradient Boosting Regressor  <br>
Deployment: Streamlit Web App <br>
</div>
""", unsafe_allow_html=True)
