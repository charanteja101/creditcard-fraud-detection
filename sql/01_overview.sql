-- Dataset overview: size, fraud count, fraud rate and money at stake.
SELECT
    COUNT(*)                                                    AS transactions,
    SUM(class)                                                  AS fraud_cases,
    ROUND(100.0 * SUM(class) / COUNT(*), 3)                     AS fraud_rate_pct,
    ROUND(SUM(amount), 0)                                       AS total_amount,
    ROUND(SUM(CASE WHEN class = 1 THEN amount ELSE 0 END), 0)   AS fraud_amount,
    ROUND(AVG(CASE WHEN class = 1 THEN amount END), 2)          AS avg_fraud_amount,
    ROUND(AVG(CASE WHEN class = 0 THEN amount END), 2)          AS avg_legit_amount
FROM transactions;
