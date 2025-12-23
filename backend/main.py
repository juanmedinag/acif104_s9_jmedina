from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import json
from pathlib import Path

app = FastAPI(title="ACIF104 Predictive Maintenance API", version="1.0")

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "gradient_boosting.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class InputData(BaseModel):
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {"detail": "metrics.json not found"}


@app.post("/predict_falla")
def predict_falla(x: InputData):
    X = np.array([[
        x.air_temperature_k,
        x.process_temperature_k,
        x.rotational_speed_rpm,
        x.torque_nm,
        x.tool_wear_min
    ]])

    Xs = scaler.transform(X)
    proba = float(model.predict_proba(Xs)[0, 1])

    # Respuesta alineada con el frontend
    return {"failure_probability": proba}
