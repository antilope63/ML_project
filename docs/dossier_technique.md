# Dossier technique — Data Pipeline Iris

## 1. Objectif
Prédire `sepal_length` à partir de `sepal_width` (régression) à partir du dataset Iris.

## 2. Dataset
- Source : `Iris Data.csv`
- Colonnes : sepal_length, sepal_width, petal_length, petal_width, species
- Cible : `sepal_length`
- Feature : `sepal_width`

## 3. Architecture
- **Stockage** : SQLite (`data/iris.db`)
- **Pipeline** : scripts Python (ingestion → SQL → train → MLflow)
- **Tracking** : MLflow local (`mlruns`)
- **API** : FastAPI (`/predict`)
- **Docker** : docker-compose (API + MLflow UI)

## 4. Pipeline détaillé
1) Ingestion CSV
2) Chargement SQL (table `iris_raw`)
3) Lecture SQL → DataFrame
4) Split train/test
5) Entraînement (LinearRegression)
6) Évaluation (MAE, RMSE, R²)
7) Logging MLflow (params, métriques, modèle)
8) Déploiement API

## 5. Choix techniques
- SQLite : simple, léger, portable
- LinearRegression : baseline explicable
- MLflow : traçabilité des runs
- FastAPI : API légère et rapide

## 6. Résultats
- MAE : ...
- RMSE : ...
- R² : ...

## 7. Limites & améliorations
- Ajouter d’autres features
- Tester d’autres modèles
- Ajouter validation croisée
- Bonus front (non réalisé)
