# Data Pipeline Iris — Régression sepal_length

Pipeline ML complet : ingestion CSV → stockage SQLite → entraînement + MLflow → API FastAPI → Docker.

## Prérequis
- Python 3.10+ (local)
- Docker + Docker Compose (pour la version dockerisée)

## Structure
- `scripts/load_db.py` : CSV → SQLite
- `scripts/train.py` : SQLite → modèle + MLflow
- `scripts/serve.py` : API FastAPI
- `data/iris.csv` : dataset
- `models/` : modèle exporté pour l’API
- `mlruns/` : tracking MLflow local

## Exécution locale
1) Installer les dépendances
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Charger le CSV dans SQLite
```bash
python scripts/load_db.py --csv "Iris Data.csv" --db data/iris.db --table iris_raw
```

3) Entraîner + log MLflow
```bash
python scripts/train.py --db data/iris.db --table iris_raw --model-out models/sepal_length_model.pkl
```

4) Lancer l’API
```bash
uvicorn scripts.serve:app --reload
```

5) Tester l’API
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"sepal_width": 3.2}'
```

6) UI MLflow (optionnel)
```bash
mlflow ui --backend-store-uri mlruns
```

## Exécution Docker
1) Build et lancer l’API + MLflow UI
```bash
docker compose up -d api mlflow
```

2) Charger le CSV (job)
```bash
docker compose run --rm load
```

3) Entraîner le modèle (job)
```bash
docker compose run --rm train
```

4) Tester l’API
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sepal_width": 3.2}'
```

5) Ouvrir MLflow UI
- http://localhost:5000

## Notes
- Le modèle prédit **sepal_length** à partir de **sepal_width**.
- Les métriques sont MAE, RMSE, R².
- L’API dépend d’un modèle déjà entraîné (`models/sepal_length_model.pkl`).
