import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.title("IOP Reports")

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

# Dashboard charts
st.subheader("IOP Time Series")
fig = px.line(df, x="timestamp", y="IOP", title="IOP Time Series", markers=True)
st.plotly_chart(fig, use_container_width=True)

st.subheader("IOP Statistics")
st.write(df["IOP"].describe())

st.subheader("Histogram")
fig2 = px.histogram(df, x="IOP", nbins=30, title="IOP Distribution")
st.plotly_chart(fig2, use_container_width=True)
