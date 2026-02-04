FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

ENV MODEL_PATH=/app/models/iris_species_model.pkl \
    MLFLOW_BACKEND_URI=file:/app/mlruns \
    MLFLOW_TRACKING_URI=file:/app/mlruns

EXPOSE 8000 5000

CMD ["/app/entrypoint.sh"]
