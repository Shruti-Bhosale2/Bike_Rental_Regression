{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b6924ce3-2a09-4fef-b4b2-1ed80fe53215",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-12 11:18:54.703 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.284 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Shri\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-01-12 11:18:55.286 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.291 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.293 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.295 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.297 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.300 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.302 Session state does not function when running a script without `streamlit run`\n",
      "2026-01-12 11:18:55.304 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.306 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.308 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.310 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.311 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.313 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.316 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.317 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.319 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.321 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.323 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.325 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.327 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.329 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.331 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.333 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.335 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.337 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.339 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.340 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.344 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.356 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.362 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.367 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.373 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.377 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 11:18:55.392 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import pandas as pd\n",
    "\n",
    "# Load model & preprocessor\n",
    "with open(\"bike_rental_xgb.pkl\", \"rb\") as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "with open(\"preprocessor.pkl\", \"rb\") as f:\n",
    "    preprocessor = pickle.load(f)\n",
    "\n",
    "st.title(\"🚲 Bike Rental Demand Prediction\")\n",
    "\n",
    "# User inputs\n",
    "season = st.selectbox(\"Season\", [1, 2, 3, 4])\n",
    "temp = st.slider(\"Temperature (°C)\", 0.0, 40.0, 20.0)\n",
    "humidity = st.slider(\"Humidity (%)\", 0, 100, 60)\n",
    "windspeed = st.slider(\"Windspeed\", 0.0, 1.0, 0.2)\n",
    "\n",
    "# Input dataframe\n",
    "input_df = pd.DataFrame({\n",
    "    \"season\": [season],\n",
    "    \"temp\": [temp],\n",
    "    \"humidity\": [humidity],\n",
    "    \"windspeed\": [windspeed]\n",
    "})\n",
    "\n",
    "if st.button(\"Predict\"):\n",
    "    transformed_input = preprocessor.transform(input_df)\n",
    "    prediction = model.predict(transformed_input)\n",
    "    st.success(f\"Predicted Bike Rentals: {int(prediction[0])}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "34e5400b-0351-4bed-b48b-27e9a8cd5182",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
