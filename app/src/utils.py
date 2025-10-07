
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_csv(path):
    return pd.read_csv(path, parse_dates=['timestamp'])

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def create_sequences(values, seq_len):
    X = []
    for i in range(len(values) - seq_len + 1):
        X.append(values[i:i+seq_len])
    return np.array(X)

def ensure_dirs(base):
    os.makedirs(base, exist_ok=True)
