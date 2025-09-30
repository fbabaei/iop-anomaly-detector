import os
import pandas as pd

def load_iop_data(filename="iop_data.csv"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", filename)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV file not found at {data_path}")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    return df
