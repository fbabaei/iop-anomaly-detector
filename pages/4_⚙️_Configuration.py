import streamlit as st

st.title("⚙️ Configuration")

threshold = st.slider("Set IOP Threshold", 10, 30, 22)
contamination = st.slider("Set Contamination Level", 0.01, 0.5, 0.1)

st.write(f"📌 Current Settings: Threshold={threshold}, Contamination={contamination}")
