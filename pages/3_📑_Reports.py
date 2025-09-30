import streamlit as st
from utils import load_iop_data

st.set_page_config(page_title="Reports", page_icon="📑")
st.subheader("Patient IOP Data")

df = load_iop_data()
st.dataframe(df.head(50))
st.write("Summary statistics:")
st.write(df.describe())
