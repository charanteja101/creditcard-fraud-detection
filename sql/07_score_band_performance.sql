-- Fraud rate by predicted-probability band (higher scores should mean much higher fraud rates).
SELECT
    CASE
        WHEN fraud_probability < 0.01 THEN '1. Under 1%'
        WHEN fraud_probability < 0.10 THEN '2. 1-10%'
        WHEN fraud_probability < 0.50 THEN '3. 10-50%'
        ELSE                               '4. 50%+'
    END                                                     AS score_band,
    COUNT(*)                                                AS transactions,
    SUM(actual_fraud)                                       AS fraud_cases,
    ROUND(100.0 * SUM(actual_fraud) / COUNT(*), 3)          AS fraud_rate_pct
FROM scored_transactions
GROUP BY score_band
ORDER BY score_band;
