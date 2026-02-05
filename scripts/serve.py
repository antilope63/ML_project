import os
from pathlib import Path
from typing import Dict, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/iris_species_model.pkl"))

app = FastAPI(title="Iris Sepal Length API", version="1.0")


class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictResponse(BaseModel):
    species_pred: str
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    decision: Optional[str] = None


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Modele introuvable: {path}. Lance scripts/train.py d'abord."
        )
    return joblib.load(path)


@app.on_event("startup")
def startup_event():
    app.state.model = load_model(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = [
        [
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ]
    ]
    model = app.state.model
    prediction = model.predict(features)[0]
    probabilities = None
    confidence = None
    decision = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        classes = model.classes_ if hasattr(model, "classes_") else range(len(proba))
        probabilities = {str(cls): float(p) for cls, p in zip(classes, proba)}
        best_idx = int(proba.argmax())
        confidence = float(proba[best_idx])
        decision = f"Je prédis {classes[best_idx]} (p={confidence:.4f})."

    return PredictResponse(
        species_pred=str(prediction),
        probabilities=probabilities,
        confidence=confidence,
        decision=decision,
    )
