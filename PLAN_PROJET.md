# Plan de projet — Data Pipeline Iris (régression sepal_length)

## Objectif

Construire un pipeline ML end‑to‑end pour prédire **`sepal_length`** à partir de **`sepal_width`**, avec stockage SQL, traçabilité MLflow, API de prédiction, et dockerisation complète.

## Livrables

1. Pipeline dockerisé (exécutable de bout en bout)
2. Dossier technique (choix, architecture, résultats)
3. Présentation / soutenance (vendredi 6 février 2026, après‑midi)

## Architecture cible (validée)

- **Data store** : SQLite
- **Training pipeline** : scripts Python (ingestion → SQL → train → évaluation → MLflow)
- **Tracking** : MLflow en stockage local (`mlruns`)
- **API** : FastAPI (obligatoire)
- **Docker** : images + docker‑compose (API + MLflow local + SQLite)

## Pipeline (ordre exact)

1. **Ingestion CSV**
   - Lire `Iris Data.csv`
   - Vérifier colonnes, types, valeurs manquantes
2. **Chargement SQL**
   - Créer table `iris_raw` (ou `iris`)
   - Insérer toutes les lignes du CSV
3. **Lecture SQL → DataFrame**
   - Lire depuis SQL pour la traçabilité
4. **Préparation**
   - `X = sepal_width` / `y = sepal_length`
   - Split train/test
5. **Entraînement**
   - Modèle de régression (baseline : LinearRegression)
6. **Évaluation**
   - MAE, RMSE, R²
7. **MLflow**
   - Log des paramètres, métriques, modèle
8. **Packaging**
   - Sauvegarde modèle pour l’API

## API (obligatoire)

- Endpoint `POST /predict`
- Entrée : JSON `{ "sepal_width": <float> }`
- Sortie : JSON `{ "sepal_length_pred": <float> }`
- Chargement du modèle MLflow ou fichier sérialisé

## Dockerisation

- **Dockerfile** pour l’app ML (train + API)
- **docker-compose.yml** pour :
  - MLflow (stockage local volume)
  - API

## Structure de fichiers (retenue)

- `data/` (optionnel si CSV reste à la racine)
- `scripts/`
  - `load_db.py` (CSV → SQL)
  - `train.py` (SQL → modèle + MLflow)
  - `serve.py` (API)
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`
- `README.md` (exécution locale + docker)
- `docs/`
  - `dossier_technique.md`
  - `presentation.md`

## Dossier technique (contenu)

- Objectif & scope
- Dataset (schéma + stats)
- Choix techniques (SQL, modèle, métriques, MLflow)
- Architecture & pipeline (diagramme)
- Résultats (métriques + discussion)
- Limites + pistes d’amélioration

## Points ouverts (restants)

- Besoin d’un `predict` CLI en plus de l’API ?
- Bonus front : type d’interface souhaitée (Streamlit, page HTML simple, autre) ?

## Planning rapide

- Jour 1 : ingestion + SQL + train
- Jour 2 : MLflow + API
- Jour 3 : Docker + docs
- Jour 4 : préparation soutenance
