
import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from joblib import load
import numpy as np
import pandas as pd
import tensorflow as tf
from src.utils import create_sequences, load_csv

DATA_PATH = os.environ.get('DATA_PATH', '/app/data/latest.csv')
MODEL_DIR = os.environ.get('MODEL_DIR', '/app/model')
SEQ_LEN = int(os.environ.get('SEQ_LEN', 10))
POLL_SECONDS = int(os.environ.get('POLL_SECONDS', 5))

app = FastAPI()
clients = set()

model = None
scaler = None
threshold = None

def load_artifacts():
    global model, scaler, threshold
    scaler = load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'lstm_autoencoder'))
    with open(os.path.join(MODEL_DIR, 'threshold.json')) as f:
        threshold = json.load(f)['threshold']
    print("Loaded model, scaler, threshold:", threshold)

@app.on_event("startup")
async def startup_event():
    try:
        load_artifacts()
    except Exception as e:
        print("Could not load artifacts on startup:", e)
    asyncio.create_task(poll_loop())

@app.get("/")
async def index():
    return HTMLResponse(open("/app/web/index.html").read())

@app.get("/anomalies")
async def anomalies():
    df = load_csv(DATA_PATH)
    features = ['cpu_usage', 'io_rate', 'mem_usage', 'job_count']
    if df.shape[0] < SEQ_LEN:
        return {"anomalies": []}
    data = df[features].values.astype('float32')
    data_scaled = scaler.transform(data)
    X = create_sequences(data_scaled, SEQ_LEN)
    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=(1,2))
    ts = df['timestamp'].iloc[SEQ_LEN-1:].astype(str).tolist()
    anomalies = []
    for i, m in enumerate(mse):
        if m > threshold:
            anomalies.append({"timestamp": ts[i], "recon_error": float(m), "index": i})
    return {"anomalies": anomalies}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        clients.remove(websocket)

async def broadcast(msg: dict):
    remove = set()
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except Exception:
            remove.add(ws)
    for r in remove:
        clients.remove(r)

async def poll_loop():
    seen = set()
    while True:
        try:
            if not os.path.exists(DATA_PATH):
                await asyncio.sleep(POLL_SECONDS)
                continue
            df = load_csv(DATA_PATH)
            features = ['cpu_usage', 'io_rate', 'mem_usage', 'job_count']
            if df.shape[0] >= SEQ_LEN:
                data = df[features].values.astype('float32')
                data_scaled = scaler.transform(data)
                X = create_sequences(data_scaled, SEQ_LEN)
                X_pred = model.predict(X)
                mse = np.mean(np.power(X - X_pred, 2), axis=(1,2))
                ts = df['timestamp'].iloc[SEQ_LEN-1:].astype(str).tolist()
                for i, m in enumerate(mse):
                    if m > threshold:
                        key = f"{ts[i]}:{round(float(m),6)}"
                        if key not in seen:
                            seen.add(key)
                            msg = {"type":"anomaly", "timestamp": ts[i], "recon_error": float(m)}
                            await broadcast(msg)
        except Exception as exc:
            print("Poll error:", exc)
        await asyncio.sleep(POLL_SECONDS)
