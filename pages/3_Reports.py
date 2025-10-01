import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="📑 Reports", layout="wide")

DATA_FILE = os.path.join("data", "iop_data.csv")
MODEL_FILE = "lstm_model.h5"
SCALER_FILE = "scaler.pkl"
LOOK_BACK = 10

# ------------------------------
# Load or generate dataset
# ------------------------------
def load_dataset():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        # Synthetic data
        np.random.seed(42)
        timestamps = pd.date_range("2023-01-01", periods=500, freq="H")
        iop_values = np.sin(np.linspace(0, 50, 500)) + np.random.normal(0, 0.3, 500)
        df = pd.DataFrame({"timestamp": timestamps, "IOP": iop_values})
    return df

# ------------------------------
# Prepare input for prediction
# ------------------------------
def prepare_input(series, look_back=LOOK_BACK):
    X = []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
    return np.array(X)

# ------------------------------
# Load model and scaler
# ------------------------------
def load_model_and_scaler():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = load_model(MODEL_FILE, compile=False)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    else:
        st.warning("⚠️ Trained model not found. Reports will only show raw stats.")
        return None, None

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📑 Reports & Analysis")

df = load_dataset()
st.subheader("📊 Raw Data Sample")
st.dataframe(df.head())

# ------------------------------
# Basic statistics
# ------------------------------
st.subheader("📈 Statistics")
st.write(df["IOP"].describe())

fig, ax = plt.subplots()
ax.plot(df["timestamp"], df["IOP"], label="IOP values")
ax.set_xlabel("Time")
ax.set_ylabel("IOP")
ax.legend()
st.pyplot(fig)

# ------------------------------
# Predictions (if model exists)
# ------------------------------
model, scaler = load_model_and_scaler()

if model and scaler:
    st.subheader("🤖 Model Predictions")

    values = df["IOP"].values.reshape(-1, 1)
    scaled = scaler.transform(values)

    X = prepare_input(scaled)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # Align timestamps
    pred_time = df["timestamp"].iloc[LOOK_BACK:].reset_index(drop=True)

    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["IOP"], label="Actual IOP")
    ax.plot(pred_time, predictions, label="Predicted IOP", linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("IOP")
    ax.legend()
    st.pyplot(fig)

    st.success("✅ Predictions plotted successfully!")
else:
    st.info("ℹ️ Train a model in the 📈 Model page to unlock predictions here.")
