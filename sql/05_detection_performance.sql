-- Detection performance at the model's decision threshold (needs the scored_transactions table).
SELECT
    SUM(CASE WHEN outcome = 'True positive'  THEN 1 ELSE 0 END)   AS true_positives,
    SUM(CASE WHEN outcome = 'False positive' THEN 1 ELSE 0 END)   AS false_positives,
    SUM(CASE WHEN outcome = 'False negative' THEN 1 ELSE 0 END)   AS false_negatives,
    SUM(CASE WHEN outcome = 'True negative'  THEN 1 ELSE 0 END)   AS true_negatives,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'True positive' THEN 1 ELSE 0 END)
          / NULLIF(SUM(actual_fraud), 0), 1)                      AS recall_pct,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'True positive' THEN 1 ELSE 0 END)
          / NULLIF(SUM(flagged), 0), 1)                           AS precision_pct,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'False positive' THEN 1 ELSE 0 END)
          / NULLIF(SUM(1 - actual_fraud), 0), 3)                  AS false_positive_rate_pct
FROM scored_transactions;
