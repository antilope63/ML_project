import argparse
import os
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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


def build_models(random_state: int):
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, random_state=random_state
        ),
        "KNeighborsClassifier": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
        "SVC": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, random_state=random_state),
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
    }


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
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
    parser.add_argument("--test-size", dest="test_size", type=float, default=0.2)
    parser.add_argument("--random-state", dest="random_state", type=int, default=42)
    parser.add_argument(
        "--select-metric",
        dest="select_metric",
        choices=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        default="accuracy",
    )
    parser.add_argument(
        "--experiment",
        dest="experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "iris_species_classification"),
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    model_out = Path(args.model_out)

    df = load_from_sql(db_path, args.table)

    X = df[FEATURES]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("target", "species")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("rows", len(df))
        mlflow.log_param("select_metric", args.select_metric)
        mlflow.log_param("classes", ",".join(sorted(y.unique())))

        input_example = X_train.head(2)
        models = build_models(args.random_state)
        results = []
        best_name = None
        best_model = None
        best_metrics = None
        best_metric_value = None

        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred)

                mlflow.log_param("model", name)
                mlflow.log_metric("accuracy", metrics["accuracy"])
                mlflow.log_metric("f1_macro", metrics["f1_macro"])
                mlflow.log_metric("precision_macro", metrics["precision_macro"])
                mlflow.log_metric("recall_macro", metrics["recall_macro"])
                mlflow.sklearn.log_model(model, "model", input_example=input_example)

                results.append({"model": name, **metrics})

                metric_value = metrics[args.select_metric]
                if is_better(metric_value, best_metric_value):
                    best_metric_value = metric_value
                    best_name = name
                    best_model = model
                    best_metrics = metrics

        if best_model is None or best_metrics is None or best_name is None:
            raise RuntimeError("Aucun modele n'a pu etre entraine.")

        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_out)
        mlflow.log_artifact(str(model_out))
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_accuracy", best_metrics["accuracy"])
        mlflow.log_metric("best_f1_macro", best_metrics["f1_macro"])
        mlflow.log_metric("best_precision_macro", best_metrics["precision_macro"])
        mlflow.log_metric("best_recall_macro", best_metrics["recall_macro"])

        stats = {
            "rows": len(df),
            "test_size": args.test_size,
            "select_metric": args.select_metric,
            "best_model": best_name,
            "best_metrics": best_metrics,
            "all_results": results,
            "features": FEATURES,
        }
        mlflow.log_dict(stats, "run_stats.json")

        sorted_results = sorted(
            results, key=lambda r: r[args.select_metric], reverse=True
        )

        print("OK: entrainement termine")
        print(f"Meilleur modele: {best_name}")
        print(
            "accuracy={accuracy:.4f} f1_macro={f1_macro:.4f} "
            "precision_macro={precision_macro:.4f} recall_macro={recall_macro:.4f}".format(
                **best_metrics
            )
        )
        print("Resultats par modele:")
        for row in sorted_results:
            print(
                "- {model}: accuracy={accuracy:.4f} f1_macro={f1_macro:.4f} "
                "precision_macro={precision_macro:.4f} recall_macro={recall_macro:.4f}".format(
                    **row
                )
            )
        print(f"Modele sauvegarde: {model_out}")


if __name__ == "__main__":
    main()
