import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("📊 IOP Monitoring Dashboard")

# Load sample data
df = pd.read_csv("data/iop_data.csv")

# KPIs
avg_iop = round(df["IOP"].mean(), 2)
max_iop = round(df["IOP"].max(), 2)
alerts = (df["IOP"] > 22).sum()

col1, col2, col3 = st.columns(3)
col1.metric("Average IOP", avg_iop)
col2.metric("Max IOP", max_iop, delta="+3")
col3.metric("Alerts (>22 mmHg)", alerts)

# Time-series chart
fig = px.line(df, x="Date", y="IOP", title="IOP Time Series", markers=True)
st.plotly_chart(fig, use_container_width=True)
