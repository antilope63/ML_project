#!/usr/bin/env sh
set -e

MLFLOW_BACKEND_URI="${MLFLOW_BACKEND_URI:-file:/app/mlruns}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
API_PORT="${API_PORT:-8000}"
MODEL_PATH="${MODEL_PATH:-/app/models/iris_species_model.pkl}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Model not found at ${MODEL_PATH}." >&2
  echo "Make sure models/iris_species_model.pkl is present in the image." >&2
  exit 1
fi

mlflow ui --host 0.0.0.0 --port "${MLFLOW_PORT}" --backend-store-uri "${MLFLOW_BACKEND_URI}" &

exec uvicorn scripts.serve:app --host 0.0.0.0 --port "${API_PORT}"
