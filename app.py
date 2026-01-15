import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Title
# -------------------------------
st.title("🚲 Bike Rental Demand Prediction Dashboard")
st.markdown("Predict bike rental demand and visualize insights using ML")
st.divider()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Input Parameters")

season = st.sidebar.selectbox("Season", [1, 2, 3, 4])
yr = st.sidebar.selectbox("Year (0 = 2018, 1 = 2019)", [0, 1])
mnth = st.sidebar.slider("Month", 1, 12)
hr = st.sidebar.slider("Hour", 0, 23)

holiday = st.sidebar.selectbox("Holiday", [0, 1])
weekday = st.sidebar.selectbox("Weekday (0=Sun, 6=Sat)", [0,1,2,3,4,5,6])
workingday = st.sidebar.selectbox("Working Day", [0, 1])
weathersit = st.sidebar.selectbox("Weather Situation", [1,2,3,4])

temp = st.sidebar.slider("Temperature", 0.0, 1.0)
atemp = st.sidebar.slider("Feels Like Temp", 0.0, 1.0)
hum = st.sidebar.slider("Humidity", 0.0, 1.0)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0)

# -------------------------------
# Input DataFrame
# -------------------------------
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
st.write("Input data types:")
st.write(input_df.dtypes)

# -------------------------------
# Initialize variables
# -------------------------------
prediction = None
demand_level = None
input_df_numeric = None

# -------------------------------
# Predict Button
# -------------------------------
if st.sidebar.button("🚀 Predict Demand"):
    prediction = model.predict(input_df)[0]
    st.success(f"🚴 Predicted Bike Rentals: {int(prediction)}")

# -------------------------------
# Display Results
# -------------------------------
if prediction is not None:

    col1, col2, col3 = st.columns(3)
    col1.metric("🚴 Predicted Rentals", int(round(prediction)))
    col2.metric("📊 Demand Level", demand_level)
    col3.metric("⏰ Hour", hr)

    st.divider()

    st.subheader("📌 Input Summary")
    st.dataframe(input_df, use_container_width=True)

    # -------------------------------
    # Hourly Demand Simulation
    # -------------------------------
    st.subheader("📈 Predicted Demand Across the Day")

    hours = np.arange(24)
    simulated = []

    for h in hours:
        temp_df = input_df.copy()
        temp_df["hr"] = h

        prediction = model.predict(temp_df)[0]
        if hasattr(model, "feature_names_in_"):
            temp_df = temp_df[model.feature_names_in_]

        simulated.append(model.predict(temp_df)[0])

    fig, ax = plt.subplots()
    ax.plot(hours, simulated)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Predicted Rentals")
    ax.set_title("Hourly Demand Forecast")
    st.pyplot(fig)

    # -------------------------------
    # Actual vs Predicted (Demo)
    # -------------------------------
    st.subheader("📊 Actual vs Predicted (Demo)")

    actual = np.array(simulated) + np.random.normal(0, 20, 24)

    fig2, ax2 = plt.subplots()
    ax2.plot(hours, actual, label="Actual")
    ax2.plot(hours, simulated, label="Predicted")
    ax2.legend()
    st.pyplot(fig2)

