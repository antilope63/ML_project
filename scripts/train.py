import argparse
import os
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine

EXPECTED_COLS = {
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
}

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

SCORING_NAME = "f1_macro"
FINAL_MODEL_NAME = "LogisticRegression"
MODEL_CHOICES = [
    "all",
    "LogisticRegression",
    "RandomForestClassifier",
    "KNeighborsClassifier",
    "SVC",
    "DecisionTreeClassifier",
]


def load_from_sql(db_path: Path, table: str) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Base SQLite introuvable: {db_path}. Lance scripts/load_db.py d'abord."
        )

    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql(f"SELECT * FROM {table}", engine)

    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans la table: {sorted(missing)}")

    return df


def build_model_specs(random_state: int):
    return {
        "LogisticRegression": {
            "estimator": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000),
            ),
            "param_grid": {
                "logisticregression__C": [0.1, 1.0, 10.0],
                "logisticregression__solver": ["lbfgs"],
            },
        },
        "RandomForestClassifier": {
            "estimator": RandomForestClassifier(random_state=random_state),
            "param_grid": {
                "n_estimators": [200, 500],
                "max_depth": [None, 3, 5, 8],
                "min_samples_split": [2, 4],
                "min_samples_leaf": [1, 2],
            },
        },
        "KNeighborsClassifier": {
            "estimator": make_pipeline(StandardScaler(), KNeighborsClassifier()),
            "param_grid": {
                "kneighborsclassifier__n_neighbors": [3, 5, 7, 9],
                "kneighborsclassifier__weights": ["uniform", "distance"],
                "kneighborsclassifier__p": [1, 2],
            },
        },
        "SVC": {
            "estimator": make_pipeline(
                StandardScaler(),
                SVC(probability=True, random_state=random_state),
            ),
            "param_grid": {
                "svc__C": [0.1, 1.0, 10.0],
                "svc__gamma": ["scale", "auto"],
                "svc__kernel": ["rbf", "linear"],
            },
        },
        "DecisionTreeClassifier": {
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "param_grid": {
                "max_depth": [None, 3, 5, 8],
                "min_samples_split": [2, 4],
                "min_samples_leaf": [1, 2],
            },
        },
    }


def compute_metrics(y_true, y_pred):
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def is_better(value: float, best_value: Optional[float]) -> bool:
    if best_value is None:
        return True
    return value > best_value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train classification model to predict iris species"
    )
    parser.add_argument("--db", dest="db_path", default="data/iris.db")
    parser.add_argument("--table", dest="table", default="iris_raw")
    parser.add_argument(
        "--model-out", dest="model_out", default="models/iris_species_model.pkl"
    )
    parser.add_argument("--random-state", dest="random_state", type=int, default=42)
    parser.add_argument(
        "--refit-full",
        dest="refit_full",
        action="store_true",
        help="Refit the best model on the full dataset before saving.",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        choices=MODEL_CHOICES,
        default="all",
        help="Train a single model or all models.",
    )
    parser.add_argument("--cv", dest="cv", type=int, default=5)
    parser.add_argument(
        "--no-search",
        dest="search",
        action="store_false",
        help="Disable hyperparameter search (faster, less accurate).",
    )
    parser.add_argument(
        "--experiment",
        dest="experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "iris_species_classification"),
    )
    parser.set_defaults(search=True)
    args = parser.parse_args()

    db_path = Path(args.db_path)
    model_out = Path(args.model_out)

    df = load_from_sql(db_path, args.table)

    X = df[FEATURES]
    y = df["species"]

    X_train, y_train = X, y

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("target", "species")
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("rows", len(df))
        mlflow.log_param("select_metric", SCORING_NAME)
        mlflow.log_param("final_model", FINAL_MODEL_NAME)
        mlflow.log_param("search", args.search)
        mlflow.log_param("cv_folds", args.cv)
        mlflow.log_param("refit_full_data", args.refit_full)
        mlflow.log_param("classes", ",".join(sorted(y.unique())))

        input_example = X_train.head(2)
        model_specs = build_model_specs(args.random_state)
        if args.model_name != "all":
            model_specs = {args.model_name: model_specs[args.model_name]}
        scoring_name = SCORING_NAME
        cv = StratifiedKFold(
            n_splits=args.cv, shuffle=True, random_state=args.random_state
        )
        results = []
        trained_models = {}
        metrics_by_model = {}
        cv_scores = {}
        best_name = None
        best_model = None
        best_metrics = None
        best_metric_value = None

        for name, spec in model_specs.items():
            with mlflow.start_run(run_name=name, nested=True):
                estimator = spec["estimator"]
                cv_score = None
                best_params = None
                if args.search:
                    search = GridSearchCV(
                        estimator,
                        spec["param_grid"],
                        scoring=scoring_name,
                        cv=cv,
                        n_jobs=-1,
                    )
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    cv_score = search.best_score_
                    best_params = search.best_params_
                else:
                    model = estimator
                    model.fit(X_train, y_train)

                y_pred = model.predict(X)
                metrics = compute_metrics(y, y_pred)

                mlflow.log_param("model", name)
                mlflow.log_metric("f1_macro", metrics["f1_macro"])
                if cv_score is not None:
                    mlflow.log_metric("cv_score", cv_score)
                if best_params is not None:
                    mlflow.log_dict(best_params, "best_params.json")
                mlflow.sklearn.log_model(model, "model", input_example=input_example)

                results.append(
                    {
                        "model": name,
                        "cv_score": cv_score,
                        "best_params": best_params,
                        **metrics,
                    }
                )
                trained_models[name] = model
                metrics_by_model[name] = metrics
                cv_scores[name] = cv_score

                metric_value = cv_score if args.search else metrics["f1_macro"]
                if is_better(metric_value, best_metric_value):
                    best_metric_value = metric_value
                    best_name = name
                    best_model = model
                    best_metrics = metrics

        if args.model_name == "all" and FINAL_MODEL_NAME in trained_models:
            best_name = FINAL_MODEL_NAME
            best_model = trained_models[FINAL_MODEL_NAME]
            best_metrics = metrics_by_model[FINAL_MODEL_NAME]
            best_metric_value = (
                cv_scores[FINAL_MODEL_NAME] if args.search else best_metrics["f1_macro"]
            )

        if best_model is None or best_metrics is None or best_name is None:
            raise RuntimeError("Aucun modele n'a pu etre entraine.")

        if args.refit_full:
            best_model.fit(X, y)

        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_out)
        mlflow.log_artifact(str(model_out))
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_f1_macro", best_metrics["f1_macro"])

        stats = {
            "rows": len(df),
            "select_metric": SCORING_NAME,
            "search": args.search,
            "cv_folds": args.cv,
            "refit_full_data": args.refit_full,
            "best_model": best_name,
            "best_metrics": best_metrics,
            "all_results": results,
            "features": FEATURES,
        }
        mlflow.log_dict(stats, "run_stats.json")

        sort_key = "cv_score" if args.search else "f1_macro"
        sorted_results = sorted(results, key=lambda r: r[sort_key], reverse=True)

        print("OK: entrainement termine")
        print(f"Modele selectionne: {best_name}")
        print("f1_macro={f1_macro:.4f}".format(**best_metrics))
        print("Resultats par modele:")
        for row in sorted_results:
            if args.search and row["cv_score"] is not None:
                print(
                    "- {model}: cv_score={cv_score:.4f} f1_macro={f1_macro:.4f}".format(
                        **row
                    )
                )
            else:
                print("- {model}: f1_macro={f1_macro:.4f}".format(**row))
        print(f"Modele sauvegarde: {model_out}")


if __name__ == "__main__":
    main()
