
import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from src.utils import create_sequences, ensure_dirs, load_csv

DATA_PATH = os.environ.get('DATA_PATH', '/app/data/latest.csv')
MODEL_DIR = os.environ.get('MODEL_DIR', '/app/model')
SEQ_LEN = int(os.environ.get('SEQ_LEN', 10))
EPOCHS = int(os.environ.get('EPOCHS', 10))
BATCH = int(os.environ.get('BATCH', 32))

def build_model(n_feats):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(SEQ_LEN, n_feats), return_sequences=False))
    model.add(RepeatVector(SEQ_LEN))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_feats)))
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    ensure_dirs(MODEL_DIR)
    df = load_csv(DATA_PATH)
    features = ['cpu_usage', 'io_rate', 'mem_usage', 'job_count']
    df = df.sort_values('timestamp').dropna(subset=features)
    data = df[features].values.astype('float32')

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

    X = create_sequences(data_scaled, SEQ_LEN)
    print("Training samples:", X.shape)

    model = build_model(X.shape[2])
    model.fit(X, X, epochs=EPOCHS, batch_size=BATCH, validation_split=0.1, shuffle=False)

    model.save(os.path.join(MODEL_DIR, 'lstm_autoencoder'))
    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=(1,2))
    threshold = float(np.mean(mse) + 3 * np.std(mse))
    with open(os.path.join(MODEL_DIR, 'threshold.json'), 'w') as f:
        json.dump({'threshold': threshold}, f)
    print("Model and threshold saved. Threshold:", threshold)

if __name__ == '__main__':
    main()
