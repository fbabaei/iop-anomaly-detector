# model_utils.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ----------------------------
# Synthetic IOP Dataset Generator
# ----------------------------
def generate_iop_data(num_patients=10, days=30, seed=42):
    np.random.seed(seed)
    data = []
    start_date = datetime.today() - timedelta(days=days)

    for pid in range(1, num_patients + 1):
        timestamps = [start_date + timedelta(days=i) for i in range(days)]
        # Simulate left and right eye IOP readings
        iop_left = np.random.normal(15, 3, size=days)
        iop_right = np.random.normal(16, 3, size=days)
        iop_mean = (iop_left + iop_right) / 2

        for t, l, r, m in zip(timestamps, iop_left, iop_right, iop_mean):
            data.append({
                "patient_id": f"Patient {pid}",
                "timestamp": t,
                "iop_left": round(l, 2),
                "iop_right": round(r, 2),
                "IOP": round(m, 2)
            })

    df = pd.DataFrame(data)
    return df

# ----------------------------
# LSTM Model for IOP Forecasting
# ----------------------------
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# ----------------------------
# Forecasting Function
# ----------------------------
def forecast_future(model, data, steps=7):
    forecast = []
    current_input = data[-1].reshape((1, -1, 1))
    for _ in range(steps):
        pred = model.predict(current_input, verbose=0)
        forecast.append(pred[0,0])
        current_input = np.append(current_input[:,1:,:], [[pred]], axis=1)
    return np.array(forecast)
