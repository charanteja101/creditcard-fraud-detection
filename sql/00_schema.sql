-- Fraud Detection: database schema (SQLite syntax; also valid in PostgreSQL).
-- Load the Kaggle file with:  python sql/load_sqlite.py

DROP TABLE IF EXISTS transactions;
CREATE TABLE transactions (
    transaction_id  INTEGER PRIMARY KEY,
    time_seconds    REAL,        -- seconds since the first transaction in the dataset (about 2 days)
    v1 REAL, v2 REAL, v3 REAL, v4 REAL, v5 REAL, v6 REAL, v7 REAL, v8 REAL, v9 REAL, v10 REAL,
    v11 REAL, v12 REAL, v13 REAL, v14 REAL, v15 REAL, v16 REAL, v17 REAL, v18 REAL, v19 REAL, v20 REAL,
    v21 REAL, v22 REAL, v23 REAL, v24 REAL, v25 REAL, v26 REAL, v27 REAL, v28 REAL,   -- anonymised PCA components
    amount          REAL,
    class           INTEGER NOT NULL   -- 1 = fraud, 0 = legitimate
);

-- Optional: model scores for the held-out test set (exported from your own scoring code, see dashboard/README.md).
DROP TABLE IF EXISTS scored_transactions;
CREATE TABLE scored_transactions (
    row_id             INTEGER PRIMARY KEY,
    hour_of_day        INTEGER,
    amount             REAL,
    actual_fraud       INTEGER NOT NULL,
    fraud_probability  REAL NOT NULL,
    flagged            INTEGER NOT NULL,
    outcome            TEXT NOT NULL   -- True positive / False positive / False negative / True negative
);
