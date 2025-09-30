import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# -----------------------------
# Load model & data
# -----------------------------
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")
data = np.load("test_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# -----------------------------
# Predict
# -----------------------------
y_pred = model.predict(X_test)

# Inverse transform back to original scale
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Flatten for plotting
y_test_inv = y_test_inv.reshape(-1)
y_pred_inv = y_pred_inv.reshape(-1)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Actual IOP")
plt.plot(y_pred_inv, label="Predicted IOP")
plt.legend()
plt.title("LSTM Forecast vs Actual IOP")
plt.xlabel("Time")
plt.ylabel("IOP")
plt.show()
