from typing import Dict
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Telecom Churn Prediction API")

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "baseline_model.pkl"
TRAIN_SPLIT_PATH = Path("data/splits/train.csv")
LOG_PATH = ARTIFACTS_DIR / "prediction_logs.csv"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Baseline model not found at {MODEL_PATH}. "
        "Run src/train_baseline.py first."
    )

if not TRAIN_SPLIT_PATH.exists():
    raise RuntimeError(
        f"Train split not found at {TRAIN_SPLIT_PATH}. "
        "Run src/data_pipeline.py first."
    )

model = joblib.load(MODEL_PATH)
FEATURES = list(pd.read_csv(TRAIN_SPLIT_PATH).drop(columns=["Churn"]).columns)


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description=(
            "Flat JSON object with all model features. "
            "Feature names must match columns from data/splits/train.csv without Churn."
        ),
    )


def build_input_frame(features: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame([features])

    # Remove unknown columns and add missing ones as zeros
    df = df.reindex(columns=FEATURES, fill_value=0.0)

    # Ensure numeric dtype
    df = df.astype(float)

    return df


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features": len(FEATURES),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        start = datetime.now(timezone.utc)

        input_df = build_input_frame(req.features)

        proba = float(model.predict_proba(input_df)[0][1])
        label = int(proba >= 0.5)

        latency_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000.0

        log_row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "score": proba,
            "prediction": label,
            **input_df.iloc[0].to_dict(),
        }

        log_df = pd.DataFrame([log_row])
        if LOG_PATH.exists():
            log_df.to_csv(LOG_PATH, mode="a", header=False, index=False)
        else:
            log_df.to_csv(LOG_PATH, index=False)

        return {
            "churn_probability": proba,
            "churn": label,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))