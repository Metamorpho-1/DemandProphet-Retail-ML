import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="DemandProphet", page_icon="🛒", layout="centered")
st.title("🛒 DemandProphet ML")
st.write("Predict optimal inventory stocking levels based on environmental and temporal factors.")

# --- ML Model Setup ---
@st.cache_resource
def train_demand_model():
    np.random.seed(42)
    days = 500
    
    # fake historical data for a generic retail product
    temp = np.random.normal(20, 10, days) # temperature
    is_holiday = np.random.choice([0, 1], days, p=[0.85, 0.15])
    is_promo = np.random.choice([0, 1], days, p=[0.8, 0.2])
    
    # Base sales + more sales on holidays/promos + slight temp factor
    base_sales = 100
    sales = base_sales + (is_holiday * 150) + (is_promo * 80) - (abs(temp - 25) * 2)
    sales = np.maximum(sales + np.random.normal(0, 10, days), 0)
    
    df = pd.DataFrame({'temp': temp, 'holiday': is_holiday, 'promo': is_promo, 'sales': sales})
    
    # TODO: Connect to actual AWS/Snowflake database later
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(df[['temp', 'holiday', 'promo']], df['sales'])
    return model

model = train_demand_model()

# --- Dashboard UI ---
st.sidebar.header("📅 Next Week's Forecast Data")
input_temp = st.sidebar.slider("Expected Temp (°C)", -5, 45, 22)
input_holiday = st.sidebar.radio("Is it a Public Holiday?", ["No", "Yes"])
input_promo = st.sidebar.radio("Active Marketing Promo?", ["No", "Yes"])

# Convert text to binary for the model
holiday_val = 1 if input_holiday == "Yes" else 0
promo_val = 1 if input_promo == "Yes" else 0

# --- Prediction ---
test_df = pd.DataFrame({'temp': [input_temp], 'holiday': [holiday_val], 'promo': [promo_val]})
predicted_demand = model.predict(test_df)[0]

st.divider()
st.subheader("Inventory Recommendation")
st.metric(label="Predicted Units Needed", value=f"{int(predicted_demand)} Units")

if holiday_val == 1 and promo_val == 1:
    st.success("📈 **High Demand Alert:** Holiday + Promo compounding effect detected. Increase logistics capacity.")
