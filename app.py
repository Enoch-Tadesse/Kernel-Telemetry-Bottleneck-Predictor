from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(
        title="Kernel Telemetry Bottleneck Predictor",
        description="Predicts server saturation using Logistic Regression trained on 8100+ real OS telemetry snapshots.",
        version="1.0"
        )

model = joblib.load('cpu_model.pkl')
scaler = joblib.load('scaler.pkl')

class KernelTelemetry(BaseModel):
        scall: float = Field(..., description="System calls per second (e.g., 2500)")
        fork: float = Field(..., description="Process forks per second (e.g., 4.0)")
        vflt: float = Field(..., description="Virtual memory page faults per second (e.g., 150)")
        runqsz: float = Field(..., description="CPU run queue size (e.g., 5.0)")
        freemem: float = Field(..., description="Free memory pages (e.g., 1200)")

@app.post("/predict")
def predict_bottleneck(data: KernelTelemetry):
        try:
                features = np.array([[data.scall, data.fork, data.vflt, data.runqsz, data.freemem]])
                scaled_features = scaler.transform(features)

                prediction = int(model.predict(scaled_features)[0])
                probability = float(model.predict_proba(scaled_features)[0][1])

                return {
                        "bottleneck_detected": bool(prediction),
                        "bottleneck_probability": round(probability, 4),
                        "status": "CRITICAL: Kernel/CPU Bottleneck" if prediction == 1 else "HEALTHY: Normal Load"
                }
        except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
        import uvicorn
        uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)