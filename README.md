# Data Pipeline Iris — Classification des espèces

Pipeline ML complet : ingestion CSV → stockage SQLite → entraînement + MLflow → API FastAPI → Docker.

## Prérequis

- Python 3.10+ (local)
- Docker (pour la version dockerisée)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer l'API (local)

```bash
uvicorn scripts.serve:app --host 0.0.0.0 --port 8000
```

## Lancer le front Streamlit

```bash
streamlit run streamlit_app.py
```

Ouvre http://localhost:8501 (l'API doit etre lancee).

## Tester l'API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## Docker (API + MLflow UI)

```bash
docker build -t iris-api .
docker run --rm -p 8000:8000 -p 5000:5000 iris-api
```

Si le port 5000 est deja utilise :

```bash
docker run --rm -p 8000:8000 -p 5001:5000 iris-api
```

MLflow UI :
- http://localhost:5000 (ou http://localhost:5001 si port change)

## Notes

- Le modele predit l'espece a partir de sepal_length, sepal_width, petal_length, petal_width.
- La reponse /predict inclut probabilities, confidence et decision.
