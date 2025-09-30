import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

DATA_FILE = "data/iop_data.csv"
MODEL_FILE = "lstm_model.h5"
LOOK_BACK = 10

# -------------------------
# Helpers
# -------------------------
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"Data file not found at {DATA_FILE}")
        st.stop()
    return pd.read_csv(DATA_FILE)

def create_dataset(series, look_back=1):
    X, Y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:(i + look_back), 0])
        Y.append(series[i + look_back, 0])
    return np.array(X), np.array(Y)

# -------------------------
# Streamlit App
# -------------------------
st.title("📑 IOP Model Reports")

if not os.path.exists(MODEL_FILE):
    st.error("No trained model found. Please train it first in the Model page.")
    st.stop()

df = load_data()
values = df["IOP"].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

X, y = create_dataset(scaled, LOOK_BACK)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = load_model(MODEL_FILE, compile=False)  # FIX

predictions = model.predict(X)
pred_inv = scaler.inverse_transform(predictions)
y_inv = scaler.inverse_transform(y.reshape(-1, 1))

# -------------------------
# Plot results
# -------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index[LOOK_BACK:], y_inv, label="Actual IOP", color="blue")
ax.plot(df.index[LOOK_BACK:], pred_inv, label="Predicted IOP", color="red")
ax.set_title("IOP Forecast vs Actual")
ax.set_xlabel("Time Index")
ax.set_ylabel("IOP")
ax.legend()

st.pyplot(fig)
