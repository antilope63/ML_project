import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

EXPECTED_COLS = {
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
}


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train regression model for sepal_length"
    )
    parser.add_argument("--db", dest="db_path", default="data/iris.db")
    parser.add_argument("--table", dest="table", default="iris_raw")
    parser.add_argument(
        "--model-out", dest="model_out", default="models/sepal_length_model.pkl"
    )
    parser.add_argument("--test-size", dest="test_size", type=float, default=0.2)
    parser.add_argument("--random-state", dest="random_state", type=int, default=42)
    parser.add_argument(
        "--experiment",
        dest="experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "iris_sepal_length"),
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    model_out = Path(args.model_out)

    df = load_from_sql(db_path, args.table)

    X = df[["sepal_width"]]
    y = df["sepal_length"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**0.5
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("feature", "sepal_width")
        mlflow.log_param("target", "sepal_length")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("rows", len(df))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        input_example = X_train.head(2)
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_out)
        mlflow.log_artifact(str(model_out))

        stats = {
            "rows": len(df),
            "test_size": args.test_size,
            "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        }
        mlflow.log_dict(stats, "run_stats.json")

        print("OK: entrainement termine")
        print(f"MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")
        print(f"Modele sauvegarde: {model_out}")


if __name__ == "__main__":
    main()
