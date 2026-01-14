import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="wide"
)

# -------------------------------
# Load Trained Model (Pipeline)
# -------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# App Title
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
weekday = st.sidebar.selectbox("Weekday (0=Sun, 6=Sat)", [0, 1, 2, 3, 4, 5, 6])
workingday = st.sidebar.selectbox("Working Day", [0, 1])
weathersit = st.sidebar.selectbox("Weather Situation", [1, 2, 3, 4])

temp = st.sidebar.slider("Temperature (normalized)", 0.0, 1.0)
atemp = st.sidebar.slider("Feels Like Temperature", 0.0, 1.0)
hum = st.sidebar.slider("Humidity", 0.0, 1.0)
windspeed = st.sidebar.slider("Windspeed", 0.0, 1.0)

# -------------------------------
# Create Input DataFrame
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

required_cols = model.feature_names_in_

missing = set(required_cols) - set(input_df_numeric.columns)

if missing:
    st.error(f"Missing columns: {missing}")
else:
    prediction = model.predict(input_df_numeric)[0]


# -------------------------------
# Initialize Prediction Variable
# -------------------------------
prediction = None
demand_level = None

# -------------------------------
# Predict Button
# -------------------------------
if st.sidebar.button("🚀 Predict Demand"):

    # ---- FORCE NUMERIC (CRITICAL FIX) ----
    input_df_numeric = input_df.apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    # ---- Ensure column order matches training ----
    if hasattr(model, "feature_names_in_"):
        input_df_numeric = input_df_numeric[model.feature_names_in_]

    try:
        prediction = float(model.predict(input_df_numeric)[0])

        # Demand category
        if prediction < 100:
            demand_level = "Low"
        elif prediction < 300:
            demand_level = "Medium"
        else:
            demand_level = "High"

        st.success("✅ Prediction generated successfully")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)

# -------------------------------
# DISPLAY RESULTS (ONLY IF EXISTS)
# -------------------------------
if prediction is not None:

    # KPI Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("🚴 Predicted Rentals", int(round(prediction)))
    col2.metric("📊 Demand Level", demand_level)
    col3.metric("⏰ Hour", hr)

    st.divider()

    # Input Summary
    st.subheader("📌 Input Feature Summary")
    st.dataframe(input_df, use_container_width=True)

    # -------------------------------
    # Demand Trend Simulation
    # -------------------------------
    st.subheader("📈 Predicted Demand Across the Day")

    hours = np.arange(0, 24)
    simulated_demand = []

    for h in hours:
        temp_df = input_df.copy()
        temp_df["hr"] = h

        temp_df = temp_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        if hasattr(model, "feature_names_in_"):
            temp_df = temp_df[model.feature_names_in_]

        simulated_demand.append(model.predict(temp_df)[0])

    fig, ax = plt.subplots()
    ax.plot(hours, simulated_demand)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Bike Rentals")
    ax.set_title("Hourly Bike Rental Demand")

    st.pyplot(fig)

    # -------------------------------
    # Actual vs Predicted (Demo)
    # -------------------------------
    st.subheader("📊 Actual vs Predicted (Sample Visualization)")

    actual_demand = np.array(simulated_demand) + np.random.normal(0, 20, size=24)

    fig2, ax2 = plt.subplots()
    ax2.plot(hours, actual_demand, label="Actual Demand")
    ax2.plot(hours, simulated_demand, label="Predicted Demand")
    ax2.legend()
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Bike Rentals")

    st.pyplot(fig2)


if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"🚴 Predicted Bike Rentals: {int(prediction[0])}")

