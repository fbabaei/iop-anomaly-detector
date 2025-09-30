# 2_📈_Model.py
import streamlit as st
import pandas as pd
import numpy as np
from model_utils import generate_iop_data, create_lstm_model, forecast_future

st.title("📈 IOP Forecasting Model")

# Generate synthetic data
df = generate_iop_data(num_patients=10, days=30)

# Select patient
patient = st.selectbox("Select Patient", df["patient_id"].unique())
patient_df = df[df["patient_id"] == patient].sort_values("timestamp")

# Show recent IOP data
st.subheader(f"Recent IOP readings for {patient}")
st.line_chart(patient_df.set_index("timestamp")["IOP"])

# Prepare data for LSTM (simple example using mean IOP)
iop_values = patient_df["IOP"].values.reshape(-1,1)
input_seq_len = 5  # last 5 days to predict next
X, y = [], []

for i in range(len(iop_values)-input_seq_len):
    X.append(iop_values[i:i+input_seq_len])
    y.append(iop_values[i+input_seq_len])

X, y = np.array(X), np.array(y)

# Train a simple LSTM
model = create_lstm_model((input_seq_len, 1))
model.fit(X, y, epochs=10, batch_size=1, verbose=0)

# Forecast next 7 days
forecast = forecast_future(model, X[-1], steps=7)
st.subheader("Forecasted IOP for next 7 days")
forecast_df = pd.DataFrame({
    "timestamp": pd.date_range(start=patient_df["timestamp"].max() + pd.Timedelta(days=1), periods=7),
    "IOP Forecast": forecast
})
st.line_chart(forecast_df.set_index("timestamp")["IOP Forecast"])
