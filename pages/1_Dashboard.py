import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="IOP Dashboard", layout="wide")
st.title("IOP Dashboard Overview")

# Paths
DATA_FILE = os.path.join("data", "iop_data.csv")

# Load or generate data
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    st.warning("CSV file not found. Generating synthetic dataset...")
    timestamps = pd.date_range("2023-01-01", periods=500, freq="H")
    iop_values = np.sin(np.linspace(0, 50, 500)) + np.random.normal(0, 0.3, 500)
    df = pd.DataFrame({"timestamp": timestamps, "IOP": iop_values})

# KPI metrics
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Average IOP", f"{df['IOP'].mean():.2f}")
col2.metric("Max IOP", f"{df['IOP'].max():.2f}")
col3.metric("Min IOP", f"{df['IOP'].min():.2f}")

# Time series chart
st.subheader("IOP Time Series")
fig_time = px.line(df, x="timestamp", y="IOP", title="IOP Over Time", markers=True)
st.plotly_chart(fig_time, use_container_width=True)

# Distribution chart
st.subheader("IOP Distribution")
fig_hist = px.histogram(df, x="IOP", nbins=30, title="IOP Value Distribution")
st.plotly_chart(fig_hist, use_container_width=True)

# Optional: rolling average
st.subheader("Rolling Average (Window=10)")
df["IOP_Rolling"] = df["IOP"].rolling(window=10).mean()
fig_roll = px.line(df, x="timestamp", y="IOP_Rolling", title="IOP Rolling Average", markers=True)
st.plotly_chart(fig_roll, use_container_width=True)
