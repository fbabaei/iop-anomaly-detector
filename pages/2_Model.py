import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import joblib

st.set_page_config(page_title="📈 LSTM Model Training", layout="wide")

DATA_FILE = "data/iop_data.csv"
MODEL_FILE = "lstm_model.h5"
SCALER_FILE = "scaler.pkl"
LOOK_BACK = 10

# ------------------------------
# Load or generate dataset
# ------------------------------
def load_dataset():
    if os.path.exists(DATA_FILE):
        st.success(f"Loaded dataset: {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
    else:
        st.warning(f"⚠️ {DATA_FILE} not found. Using synthetic dataset instead.")
        np.random.seed(42)
        time = pd.date_range("2023-01-01", periods=500, freq="H")
        sensor_values = np.sin(np.linspace(0, 50, 500)) + np.random.normal(0, 0.3, 500)
        df = pd.DataFrame({"timestamp": time, "IOP": sensor_values})
    return df

# ------------------------------
# Prepare LSTM data
# ------------------------------
def create_dataset(series, look_back=LOOK_BACK):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

# ------------------------------
# Train and save model
# ------------------------------
def train_and_save():
    df = load_dataset()
    values = df["IOP"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    X, y = create_dataset(scaled)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(LOOK_BACK, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=16, verbose=1)

    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    st.success("✅ Model trained and saved successfully!")

# ------------------------------
# Load existing model
# ------------------------------
def load_existing_model():
    if os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE, compile=False)  # avoid 'mse' error
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    else:
        st.error("❌ No trained model found. Please train first.")
        return None, None

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📈 LSTM Model Training")

if st.button("Train New Model"):
    train_and_save()

if st.button("Load Existing Model"):
    model, scaler = load_existing_model()
    if model:
        st.success("✅ Model loaded successfully!")
