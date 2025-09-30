import pandas as pd
import numpy as np
import os

# Make sure the data folder exists
os.makedirs("data", exist_ok=True)

# Parameters
n_patients = 10
n_days = 30
measurements_per_day = 2  # morning + evening
total_rows = n_patients * n_days * measurements_per_day

# Generate patient IDs
patient_ids = [f"P{str(i).zfill(3)}" for i in range(1, n_patients+1)]

# Generate timestamps
date_range = pd.date_range("2025-09-01", periods=n_days, freq="D")
timestamps = []
for d in date_range:
    timestamps.append(d.replace(hour=8))   # morning
    timestamps.append(d.replace(hour=20))  # evening

timestamps = timestamps * n_patients
patient_column = np.repeat(patient_ids, n_days * measurements_per_day)

# Generate synthetic IOP readings
np.random.seed(42)
iop_left = np.random.normal(loc=16, scale=2, size=total_rows).clip(10, 30).round()
iop_right = np.random.normal(loc=16, scale=2, size=total_rows).clip(10, 30).round()

# Build DataFrame
df = pd.DataFrame({
    "patient_id": patient_column,
    "timestamp": timestamps,
    "iop_left": iop_left,
    "iop_right": iop_right
})

# Add combined IOP column (average of left and right)
df['IOP'] = df[['iop_left', 'iop_right']].mean(axis=1).round(1)

# Save to CSV
df.to_csv("data/iop_data.csv", index=False)

print("Synthetic dataset saved to data/iop_data.csv")
print(df.head(10))
