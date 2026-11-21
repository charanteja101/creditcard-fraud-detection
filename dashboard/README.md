# Fraud dashboard

Two report pages built from the scored held-out test set (56,746 transactions, 95 fraud cases).

> **Ready-made dashboard images:** `python dashboard/build_dashboard.py` renders the two pages below as PNG files in `dashboard/screenshots/`, using the same bands, outcome logic and measures as `measures.dax`. They are generated with matplotlib, not exported from Power BI. The steps further down explain how to build the interactive Power BI version, because a `.pbix` file cannot be generated from code.

## 1. Export the scored test set

`python -m src.train` writes `dashboard/powerbi_data/scored_test.csv` automatically. To rebuild it from the saved model without retraining:

```bash
python -m src.export_scores --data data/creditcard.csv
```

The folder is git-ignored. The file has one row per held-out transaction and these columns:

| Column | Meaning |
|---|---|
| `row_id` | Row index of the de-duplicated dataset (unique) |
| `hour_of_day` | Integer 0-23, `floor((Time % 86400) / 3600)`, relative to the start of the two-day window |
| `amount` | Transaction amount |
| `actual_fraud` | 1 if the transaction was fraud, else 0 |
| `fraud_probability` | Predicted probability from the hybrid model |
| `flagged` | 1 if the probability is at or above the decision threshold (0.112), else 0 |
| `outcome` | True positive / False positive / False negative / True negative |

## 2. Generate the dashboard images

```bash
python dashboard/build_dashboard.py
```

| Image | Content |
|---|---|
| `page1_fraud_overview.png` | Total transactions and fraud cases, fraud cases by hour, fraud rate by amount band |
| `page2_detection_performance.png` | Recall, precision, false-positive rate, prediction outcomes, fraud amount caught vs missed vs legitimate amount blocked |

Fraud counts per hour and per amount band are small (95 cases in total), so treat those charts as indicative.

## 3. Build the interactive report in Power BI Desktop (optional)

1. Get Data -> Text/CSV -> load `scored_test.csv` and name the table `scored_test`.
2. Add the calculated columns and measures from `measures.dax`. The band labels start with a number so Power BI sorts them in the right order.
3. Suggested layout:
   - **Page 1, Fraud overview:** cards for Total Transactions, Actual Fraud Cases, Fraud Rate and Flagged Transactions; a column chart of Actual Fraud Cases by `hour_of_day`; a column chart of Fraud Rate by `amount_band`.
   - **Page 2, Detection performance:** cards for Recall, Precision and False Positive Rate; a bar chart of `outcome`; cards for Fraud Amount Caught, Fraud Amount Missed and Legit Amount Blocked.
   - Slicers: `amount_band`, `score_band`, `hour_of_day`.
4. Save as `dashboard/fraud_dashboard.pbix` and export page images to `dashboard/screenshots/`.

The DAX has not been tested in Power BI Desktop, so check that the column and table names match your CSV.
