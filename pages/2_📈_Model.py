import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------
# Parameters
# -----------------------------
N_STEPS = 30     # lookback window
N_FEATURES = 1   # univariate time series

# -----------------------------
# Load & Preprocess
# -----------------------------
df = pd.read_csv("iop_data.csv")  # make sure this file exists
series = df["IOP"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(series)

# Sliding window generator
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, N_STEPS)

# Reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], N_STEPS, N_FEATURES))
y = y.reshape((y.shape[0], 1))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -----------------------------
# Build Model
# -----------------------------
model = Sequential([
    LSTM(50, activation="relu", input_shape=(N_STEPS, N_FEATURES)),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# -----------------------------
# Save model & artifacts
# -----------------------------
model.save("lstm_model.h5")
joblib.dump(scaler, "scaler.pkl")
np.savez("test_data.npz", X_test=X_test, y_test=y_test)
