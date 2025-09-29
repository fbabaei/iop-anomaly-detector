import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.title("📈 Anomaly Detection Model")

df = pd.read_csv("data/iop_data.csv")

# Train model
model = IsolationForest(contamination=0.1, random_state=42)
df["Anomaly"] = model.fit_predict(df[["IOP"]])

# Plot
fig, ax = plt.subplots()
ax.plot(df["Date"], df["IOP"], label="IOP")
ax.scatter(df["Date"], df["IOP"], c=df["Anomaly"].map({1:"blue", -1:"red"}), label="Anomalies")
plt.xticks(rotation=45)
ax.legend()
st.pyplot(fig)
