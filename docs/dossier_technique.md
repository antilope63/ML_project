# Dossier technique — Data Pipeline Iris

## 1. Objectif

Prédire l’**espèce** (`species`) à partir des 4 mesures (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).

## 2. Dataset

- Source : `Iris Data.csv`
- Colonnes : sepal_length, sepal_width, petal_length, petal_width, species
- Cible : `species`
- Features : sepal_length, sepal_width, petal_length, petal_width

## 3. Architecture

- **Stockage** : SQLite (`data/iris.db`)
- **Pipeline** : scripts Python (ingestion → SQL → train → MLflow)
- **Tracking** : MLflow local (`mlruns`)
- **API** : FastAPI (`/predict`)
- **Docker** : image unique (API + MLflow UI)

## 4. Pipeline détaillé

1. Ingestion CSV
2. Chargement SQL (table `iris_raw`)
3. Lecture SQL → DataFrame
4. Entraînement + comparaison de 5 modèles (classification)
5. Recherche d’hyperparamètres (CV) + sélection du meilleur
6. Évaluation via CV (f1_macro)
7. Logging MLflow (params, métriques, modèle)
8. Refit final sur 100% des données (option `--refit-full`)
9. Déploiement API

## 5. Choix techniques

- SQLite : simple, léger, portable
- Modèles : LogisticRegression, RandomForest, KNN, SVC, DecisionTree
- Sélection : f1_macro
- Validation : CV pour hyperparamètres
- Modèle exporté : LogisticRegression
- MLflow : traçabilité des runs
- FastAPI : API légère et rapide

## 6. Résultats

- F1 macro : ...

## 7. Limites & améliorations

- Ajuster la sélection et les grilles d’hyperparamètres
- Ajouter matrice de confusion et rapport de classification
- Étendre l’API avec probas et explications
- Bonus front (non réalisé)
