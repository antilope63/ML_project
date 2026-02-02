# Data Pipeline Iris — Classification des espèces

Pipeline ML complet : ingestion CSV → stockage SQLite → entraînement + MLflow → API FastAPI → Docker.

## Prérequis

- Python 3.10+ (local)
- Docker + Docker Compose (pour la version dockerisée)

## Structure

- `scripts/load_db.py` : CSV → SQLite
- `scripts/train.py` : SQLite → modèle de classification + MLflow
- `scripts/serve.py` : API FastAPI
- `data/iris.csv` : dataset
- `models/` : modèle exporté pour l’API
- `mlruns/` : tracking MLflow local

## Exécution locale

1. Installer les dépendances

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Charger le CSV dans SQLite

```bash
python scripts/load_db.py --csv "Iris Data.csv" --db data/iris.db --table iris_raw
```

3. Entraîner + comparer 5 modèles + log MLflow (classification)

```bash
python scripts/train.py --db data/iris.db --table iris_raw --model-out models/iris_species_model.pkl
```

Par défaut, le meilleur modèle est choisi selon l’**accuracy**. Tu peux changer la métrique de sélection :

```bash
python scripts/train.py --db data/iris.db --table iris_raw --model-out models/iris_species_model.pkl --select-metric accuracy
```

Options: `accuracy`, `f1_macro`, `precision_macro`, `recall_macro`.

4. Lancer l’API

```bash
uvicorn scripts.serve:app --reload
```

5. Tester l’API

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

6. UI MLflow (optionnel)

```bash
mlflow ui --backend-store-uri mlruns
```

## Exécution Docker

1. Build et lancer l’API + MLflow UI

```bash
docker compose up -d api mlflow
```

2. Charger le CSV (job)

```bash
docker compose run --rm load
```

3. Entraîner le modèle (job)

```bash
docker compose run --rm train
```

4. Tester l’API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

5. Ouvrir MLflow UI

- http://localhost:5000

## Notes

- Le modèle prédit l’**espèce** à partir de **sepal_length, sepal_width, petal_length, petal_width**.
- Les métriques sont accuracy, F1 macro, précision macro, rappel macro.
- 5 modèles sont comparés: LogisticRegression, RandomForestClassifier, KNeighborsClassifier, SVC, DecisionTreeClassifier.
- L’API dépend du **meilleur modèle** entraîné (`models/iris_species_model.pkl`).
