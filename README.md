# Kernel Telemetry Bottleneck Predictor

A production-ready machine learning microservice that predicts whether a server is experiencing a critical CPU/kernel bottleneck based on real-time operating system telemetry metrics.

---

## Problem Statement

Modern backend infrastructure demands proactive monitoring. By the time a sysadmin notices degraded performance, the damage — dropped requests, cascading timeouts, SLA breaches — is already done. This project tackles that gap by training a binary classifier on **real UNIX server workload snapshots** and exposing it through a lightweight inference API, enabling automated bottleneck detection before critical failure.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
│                                                         │
│  OpenML (cpu_act)  ──►  Feature Engineering  ──►  Model │
│  8,192 snapshots        5 OS telemetry features   Train │
│                         StandardScaler             │    │
│                         Logistic Regression        │    │
│                                                    ▼    │
│                                  cpu_model.pkl  scaler.pkl
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Inference Service                      │
│                                                         │
│  JSON Payload  ──►  Pydantic Validation  ──►  Scaling   │
│  (5 metrics)        Schema enforcement       StandardScaler
│                                                  │      │
│                                                  ▼      │
│                                         Model Predict   │
│                                          │         │    │
│                                          ▼         ▼    │
│                                       Label    Probability
│                                    (0 or 1)    (0.0–1.0) │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| ML Framework | scikit-learn (Logistic Regression) |
| Data Source | OpenML `cpu_act` benchmark (8,192 real UNIX server snapshots) |
| Feature Scaling | StandardScaler (fit on training data only) |
| API Framework | FastAPI |
| Schema Validation | Pydantic |
| Model Serialization | joblib |
| ASGI Server | Uvicorn |

## Features Used

Five core OS telemetry signals were selected as input features:

| Feature | Description |
|:--------|:------------|
| `scall` | System calls per second |
| `fork` | Process forks per second |
| `vflt` | Virtual memory page faults per second |
| `runqsz` | CPU run queue size |
| `freemem` | Free memory pages available |

**Target variable:** Binary label derived from the `usr` column — a server is classified as experiencing a **bottleneck** (`1`) when CPU user-space efficiency drops below 80% capacity.

## Project Structure

```
server-bottleneck-predictor/
├── train.py            # Trains the Logistic Regression model & saves artifacts
├── app.py              # FastAPI microservice for live inference
├── cpu_model.pkl       # Serialized trained model weights
├── scaler.pkl          # Serialized StandardScaler (fitted on training data)
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Enoch-Tadesse/Kernel-Telemetry-Bottleneck-Predictor.git
cd Kernel-Telemetry-Bottleneck-Predictor
```

### 2. Create a Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the Model (Optional — pre-trained artifacts are included)

```bash
python train.py
```

This fetches the `cpu_act` dataset from OpenML, trains a Logistic Regression classifier, and serializes `cpu_model.pkl` and `scaler.pkl` to disk.

### 4. Start the API Server

```bash
python app.py
```

The service will be available at `http://127.0.0.1:8000`.

## API Usage

### `POST /predict`

Send a JSON payload with five kernel telemetry metrics:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "scall": 2500,
    "fork": 4.0,
    "vflt": 150,
    "runqsz": 5.0,
    "freemem": 1200
  }'
```

#### Response — Healthy Server

```json
{
  "bottleneck_detected": false,
  "bottleneck_probability": 0.1823,
  "status": "HEALTHY: Normal Load"
}
```

#### Response — Bottleneck Detected

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "scall": 8000,
    "fork": 40.0,
    "vflt": 900,
    "runqsz": 25.0,
    "freemem": 50
  }'
```

```json
{
  "bottleneck_detected": true,
  "bottleneck_probability": 0.9412,
  "status": "CRITICAL: Kernel/CPU Bottleneck"
}
```

### Interactive API Docs

FastAPI auto-generates interactive documentation:

- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Engineering Decisions

- **Real-world data over toy datasets.** The OpenML `cpu_act` benchmark contains 8,192 snapshots of actual UNIX server workloads — not synthetic or academic data.
- **No data leakage.** `StandardScaler` is fit exclusively on the training split. The test set and inference pipeline only call `.transform()`.
- **Modular, production-oriented code.** Training (`train.py`) and serving (`app.py`) are cleanly separated — no Jupyter notebook artifacts in the deployment path.
- **Strict input validation.** Pydantic enforces schema constraints on every incoming request before it reaches the model.

## License

This project is open source and available under the [MIT License](LICENSE).
