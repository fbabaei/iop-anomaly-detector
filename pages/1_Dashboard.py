import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="📊 IoT Anomaly Dashboard", layout="wide")

DATA_FILE = os.path.join("data", "iop_data.csv")

# Try loading dataset
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    st.success(f"Loaded dataset: {DATA_FILE}")
else:
    st.warning(f"⚠️ {DATA_FILE} not found. Using synthetic dataset instead.")
    # Create synthetic fallback dataset
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=500, freq="H")
    iop_values = np.sin(np.linspace(0, 50, 500)) + np.random.normal(0, 0.3, 500)
    df = pd.DataFrame({"timestamp": timestamps, "IOP": iop_values})

# Show preview
st.subheader("📈 Dataset Preview")
st.dataframe(df.head())

# Basic plot
st.subheader("📊 Sensor Trends")
st.line_chart(df.set_index(df.columns[0]))
