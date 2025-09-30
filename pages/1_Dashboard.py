# 1_📊_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from model_utils import generate_iop_data

st.set_page_config(page_title="IOP Dashboard", layout="wide")
st.title("📊 IOP Dashboard")

# Generate synthetic data
df = generate_iop_data(num_patients=10, days=30)

# ------------------------
# Sidebar Filters
# ------------------------
st.sidebar.header("Filters")
selected_patient = st.sidebar.selectbox("Select Patient", df["patient_id"].unique())
date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["timestamp"].min(), df["timestamp"].max()]
)

# Filter data
filtered_df = df[
    (df["patient_id"] == selected_patient) &
    (df["timestamp"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# ------------------------
# KPIs
# ------------------------
st.subheader(f"Key Metrics for {selected_patient}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average IOP", round(filtered_df["IOP"].mean(), 2))
col2.metric("Max IOP", filtered_df["IOP"].max())
col3.metric("Min IOP", filtered_df["IOP"].min())
col4.metric("IOP Std Dev", round(filtered_df["IOP"].std(), 2))

# ------------------------
# Charts
# ------------------------
st.subheader("IOP Time Series")
fig_line = px.line(
    filtered_df,
    x="timestamp",
    y="IOP",
    title=f"IOP Trend for {selected_patient}",
    markers=True
)
st.plotly_chart(fig_line, use_container_width=True)

st.subheader("IOP Distribution")
fig_hist = px.histogram(
    filtered_df,
    x="IOP",
    nbins=15,
    title=f"IOP Distribution for {selected_patient}"
)
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Comparing Left and Right Eye IOP")
fig_scatter = px.scatter(
    filtered_df,
    x="iop_left",
    y="iop_right",
    title="Left vs Right Eye IOP",
    trendline="ols"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------
# Summary Table
# ------------------------
st.subheader("Patient Summary Table")
summary_table = filtered_df.groupby("patient_id")["IOP"].agg(["mean", "min", "max", "std"]).reset_index()
st.dataframe(summary_table)
