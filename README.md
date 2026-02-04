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

La comparaison utilise la **f1_macro** en CV.
L’entraînement se fait sur 100% des données (pas de split).
Le modèle exporté pour l’API est **LogisticRegression**.

Pour entraîner un seul modèle (ex: LogisticRegression) :

```bash
python scripts/train.py --db data/iris.db --table iris_raw --model-out models/iris_species_model.pkl --model LogisticRegression --refit-full
```

Pour sauvegarder un modèle final entraîné sur 100% des données (après sélection),
ajoute `--refit-full` :

```bash
python scripts/train.py --db data/iris.db --table iris_raw --model-out models/iris_species_model.pkl --refit-full
```

Par défaut, une recherche d’hyperparamètres (CV) est faite pour chaque modèle. Pour aller plus vite :

```bash
python scripts/train.py --db data/iris.db --table iris_raw --model-out models/iris_species_model.pkl --no-search
```

Tu peux aussi changer le nombre de folds :

```bash
python scripts/train.py --db data/iris.db --table iris_raw --model-out models/iris_species_model.pkl --cv 3
```

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
mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000
```

## Exécution Docker

1. Entraîner un seul modèle (LogisticRegression) sur 100% des données

```bash
python3 -B scripts/train.py --db data/iris.db --table iris_raw \
  --model-out models/iris_species_model.pkl --model LogisticRegression --cv 10 --refit-full
```

2. Build et lancer l’API + MLflow UI (dans un seul container)

```bash
docker build -t iris-api .
docker run --rm -p 8000:8000 -p 5000:5000 iris-api
```

Si le port 5000 est déjà utilisé, lance plutôt :

```bash
docker run --rm -p 8000:8000 -p 5001:5000 iris-api
```

3. Tester l’API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

4. Ouvrir MLflow UI

- http://localhost:5000
- (ou http://localhost:5001 si tu as changé le port)

## Notes

- Le modèle prédit l’**espèce** à partir de **sepal_length, sepal_width, petal_length, petal_width**.
- La métrique utilisée est **f1_macro**.
- 5 modèles peuvent être comparés: LogisticRegression, RandomForestClassifier, KNeighborsClassifier, SVC, DecisionTreeClassifier.
- Le modèle exporté pour l’API est **LogisticRegression** (`models/iris_species_model.pkl`).
- La réponse `/predict` inclut `probabilities`, `confidence` et `decision` (phrase explicite).
