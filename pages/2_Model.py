import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError

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
    df = pd.read_csv(DATA_FILE)
    return df

def create_dataset(series, look_back=1):
    X, Y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:(i + look_back), 0])
        Y.append(series[i + look_back, 0])
    return np.array(X), np.array(Y)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss=MeanSquaredError())
    return model

def train_and_save():
    df = load_data()
    values = df["IOP"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    X, y = create_dataset(scaled, LOOK_BACK)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = build_model((LOOK_BACK, 1))
    model.fit(X, y, epochs=5, batch_size=16, verbose=1)

    model.save(MODEL_FILE)
    return model, scaler

def load_existing_model():
    if not os.path.exists(MODEL_FILE):
        return None, None
    model = load_model(MODEL_FILE, compile=False)  # FIX: avoid "mse" issue
    df = load_data()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df["IOP"].values.reshape(-1, 1))
    return model, scaler

# -------------------------
# Streamlit App
# -------------------------
st.title("📈 IOP Forecasting Model")

model, scaler = load_existing_model()
if model is None:
    st.warning("No trained model found. Training a new one...")
    model, scaler = train_and_save()
    st.success("Model trained and saved!")

st.success("Model is ready for predictions.")
