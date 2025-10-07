
# Mainframe Deep Learning Anomaly Detector

This project demonstrates real-time anomaly detection on IBM z/OS mainframe data using Python, TensorFlow (LSTM Autoencoder), and Ansible automation running inside a zCX container.

## Quick start (local testing)
1. cd app
2. python3 src/generate_synthetic.py
3. python3 src/train.py
4. uvicorn src.infer_service:app --host 0.0.0.0 --port 8080 --reload
5. Open http://localhost:8080
