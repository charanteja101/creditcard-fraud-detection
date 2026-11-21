import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="Load creditcard.csv (and optional model scores) into SQLite")
    ap.add_argument("--data", default=str(ROOT / "data" / "creditcard.csv"))
    ap.add_argument("--scores", default=str(ROOT / "dashboard" / "powerbi_data" / "scored_test.csv"))
    ap.add_argument("--db", default=str(ROOT / "sql" / "fraud.db"))
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.executescript((ROOT / "sql" / "00_schema.sql").read_text())

    df = pd.read_csv(args.data)
    df = df.rename(columns={"Time": "time_seconds", "Amount": "amount", "Class": "class",
                            **{f"V{i}": f"v{i}" for i in range(1, 29)}})
    df.insert(0, "transaction_id", range(1, len(df) + 1))
    df.to_sql("transactions", con, if_exists="append", index=False)
    print(f"transactions: {len(df):,} rows loaded")

    if Path(args.scores).exists():
        s = pd.read_csv(args.scores)
        if "outcome" not in s.columns:
            s["outcome"] = np.select(
                [
                    (s["flagged"] == 1) & (s["actual_fraud"] == 1),
                    (s["flagged"] == 1) & (s["actual_fraud"] == 0),
                    (s["flagged"] == 0) & (s["actual_fraud"] == 1),
                ],
                ["True positive", "False positive", "False negative"],
                default="True negative",
            )
        s = s[["row_id", "hour_of_day", "amount", "actual_fraud", "fraud_probability", "flagged", "outcome"]]
        s.to_sql("scored_transactions", con, if_exists="append", index=False)
        print(f"scored_transactions: {len(s):,} rows loaded")
    else:
        print("scored_transactions: skipped (export your scored test set first, see dashboard/README.md)")

    con.commit()
    con.close()
    print(f"Database written to {args.db}")
    print("Run a query, e.g.:  sqlite3 sql/fraud.db < sql/02_fraud_by_hour.sql")


if __name__ == "__main__":
    main()
