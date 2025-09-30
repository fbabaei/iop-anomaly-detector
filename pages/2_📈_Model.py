import streamlit as st
from utils import load_iop_data
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

st.set_page_config(page_title="Model", page_icon="📈")

df = load_iop_data()
st.subheader("LSTM Forecasting Example")

# Scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["iop_left", "iop_right"]])

# Simple LSTM dataset creation
def create_sequences(data, seq_length=5):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        xs.append(data[i:(i+seq_length)])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

X, y = create_sequences(scaled)
st.write(f"Sample sequence X shape: {X.shape}, y shape: {y.shape}")

# LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1]))
model.compile(optimizer='adam', loss='mse')

st.write("Training model...")
model.fit(X, y, epochs=3, batch_size=16, verbose=0)
st.success("Model trained (demo run)")
