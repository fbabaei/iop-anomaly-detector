import streamlit as st
import pandas as pd

st.title("📑 Reports & Export")

df = pd.read_csv("data/iop_data.csv")
st.write(df.head())

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Report (CSV)", csv, "iop_report.csv", "text/csv")
