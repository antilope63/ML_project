import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/sepal_length_model.pkl"))

app = FastAPI(title="Iris Sepal Length API", version="1.0")


class PredictRequest(BaseModel):
    sepal_width: float


class PredictResponse(BaseModel):
    sepal_length_pred: float


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

    prediction = app.state.model.predict([[request.sepal_width]])
    return PredictResponse(sepal_length_pred=float(prediction[0]))
