# 3_📑_Reports.py
import streamlit as st
import pandas as pd
from model_utils import generate_iop_data

st.title("📑 IOP Reports")

# Generate synthetic data
df = generate_iop_data(num_patients=10, days=30)

# Patient-wise summary
patient_summary = df.groupby("patient_id")["IOP"].agg(["mean", "min", "max", "std"]).reset_index()
st.subheader("Patient Summary Statistics")
st.dataframe(patient_summary)

# Highlight patients with high variability
high_variability = patient_summary[patient_summary["std"] > 3]
st.subheader("Patients with High IOP Variability")
st.dataframe(high_variability)

# Export report button
st.download_button(
    label="Download Full Report as CSV",
    data=df.to_csv(index=False),
    file_name="iop_report.csv",
    mime="text/csv"
)
