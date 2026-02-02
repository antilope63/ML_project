import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine

CANDIDATE_CSVS = [
    "data/iris.csv",
    "Iris Data.csv",
    "iris.csv",
]


def resolve_csv_path(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value)

    for candidate in CANDIDATE_CSVS:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        "CSV introuvable. Utilise --csv ou place le fichier dans data/iris.csv"
    )


def load_csv_to_sql(csv_path: Path, db_path: Path, table: str) -> int:
    df = pd.read_csv(csv_path)

    expected_cols = {
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql(table, engine, if_exists="replace", index=False)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Charge le CSV Iris dans SQLite")
    parser.add_argument("--csv", dest="csv_path", default=None)
    parser.add_argument("--db", dest="db_path", default="data/iris.db")
    parser.add_argument("--table", dest="table", default="iris_raw")
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv_path)
    db_path = Path(args.db_path)

    row_count = load_csv_to_sql(csv_path, db_path, args.table)
    print(f"OK: {row_count} lignes chargees dans {db_path} (table {args.table})")


if __name__ == "__main__":
    main()
