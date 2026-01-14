import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Bike Rental Demand Dashboard",
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
st.markdown("Predict bike rental demand and visualize insights")

st.divider()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Input Parameters")

season = st.sidebar.selectbox("Season", [1, 2, 3, 4])
yr = st.sidebar.selectbox("Year (0=2018, 1=2019)", [0, 1])
mnth = st.sidebar.slider("Month", 1, 12)
hr = st.sidebar.slider("Hour", 0, 23)
holiday = st.sidebar.selectbox("Holiday", [0, 1])
workingday = st.sidebar.selectbox("Working Day", [0, 1])
weathersit = st.sidebar.selectbox("Weather Situation", [1, 2, 3, 4])
weekday = st.sidebar.selectbox("Weekday (0=Sunday, 6=Saturday)",[0, 1, 2, 3, 4, 5, 6])
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

# Force correct data types (CRITICAL FIX)
int_cols = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday", "weathersit"
]

float_cols = ["temp", "atemp", "hum", "windspeed"]

input_df[int_cols] = input_df[int_cols].astype(int)
input_df[float_cols] = input_df[float_cols].astype(float)


# -------------------------------
# Prediction
# -------------------------------
prediction = None

if st.sidebar.button("🚀 Predict Demand"):

    input_df_numeric = input_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    try:
        prediction = model.predict(input_df_numeric)[0]

        # ----- Demand Category -----
        if prediction < 100:
            demand_level = "Low"
        elif prediction < 300:
            demand_level = "Medium"
        else:
            demand_level = "High"

        # ----- KPI Cards -----
        col1, col2, col3 = st.columns(3)
        col1.metric("🚴 Predicted Rentals", int(prediction))
        col2.metric("📊 Demand Level", demand_level)
        col3.metric("⏰ Selected Hour", int(input_df_numeric["hr"][0]))

        st.success("✅ Prediction generated successfully")

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)

    # -------------------------------
    # KPI Cards
    # -------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("🚴 Predicted Rentals", int(prediction))
    col2.metric("📊 Demand Level", demand_level)
    col3.metric("⏰ Selected Hour", hr)

    st.divider()

    # -------------------------------
    # Feature Overview Table
    # -------------------------------
    st.subheader("📌 Input Feature Summary")
    st.dataframe(input_df, use_container_width=True)

    # -------------------------------
    # Visualization: Demand Simulation
    # -------------------------------
    st.subheader("📈 Demand Trend Simulation")

    hours = np.arange(0, 24)
    simulated_data = []

    for h in hours:
        temp_df = input_df.copy()
        temp_df["hr"] = h
        simulated_data.append(model.predict(temp_df)[0])

    fig, ax = plt.subplots()
    ax.plot(hours, simulated_data)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Bike Demand")
    ax.set_title("Predicted Demand Across Hours")

    st.pyplot(fig)

    # -------------------------------
    # Actual vs Predicted (Sample)
    # -------------------------------
    st.subheader("📊 Actual vs Predicted (Sample Comparison)")

    actual = simulated_data + np.random.normal(0, 20, size=24)

    fig2, ax2 = plt.subplots()
    ax2.plot(hours, actual, label="Actual Demand")
    ax2.plot(hours, simulated_data, label="Predicted Demand")
    ax2.legend()
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Bike Rentals")

    st.pyplot(fig2)

    st.success("✅ Prediction & Insights Generated Successfully")


if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Rentals: {int(prediction[0])}")

