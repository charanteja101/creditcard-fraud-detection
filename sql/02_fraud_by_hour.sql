-- Fraud rate by hour of day (time_seconds counts from the first transaction, so hour 0 is the start of the capture window).
SELECT
    CAST(time_seconds AS INTEGER) % 86400 / 3600    AS hour_of_day,
    COUNT(*)                                        AS transactions,
    SUM(class)                                      AS fraud_cases,
    ROUND(100.0 * SUM(class) / COUNT(*), 3)         AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;
