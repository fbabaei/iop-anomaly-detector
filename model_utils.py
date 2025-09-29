import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def generate_iot_data(n=200, anomaly_frac=0.05):
    """Generate synthetic IoT sensor readings with anomalies."""
    time = np.arange(n)
    sensor = np.sin(time/10) + np.random.normal(0, 0.1, size=n)

    # Inject anomalies
    anomalies = np.random.choice(n, int(n * anomaly_frac), replace=False)
    sensor[anomalies] += np.random.uniform(3, 5, size=len(anomalies))  # spikes

    df = pd.DataFrame({"time": time, "sensor": sensor})
    return df, anomalies

def detect_anomalies(df):
    """Detect anomalies using Isolation Forest."""
    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(df[['sensor']])
    df['anomaly'] = preds == -1
    return df
