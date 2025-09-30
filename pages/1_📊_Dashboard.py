import streamlit as st
import plotly.express as px
from utils import load_iop_data

st.set_page_config(page_title="Dashboard", page_icon="📊")

df = load_iop_data()

# KPIs
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Patients", df["patient_id"].nunique())
col2.metric("Avg IOP Left", round(df["iop_left"].mean(), 2))
col3.metric("Avg IOP Right", round(df["iop_right"].mean(), 2))

# Line chart
st.subheader("IOP Time Series")
fig = px.line(df, x="timestamp", y=["iop_left", "iop_right"], markers=True,
              labels={"value":"IOP", "timestamp":"Time"}, title="IOP over Time")
st.plotly_chart(fig, use_container_width=True)
