-- Fraud rate and fraud money by transaction amount band.
SELECT
    CASE
        WHEN amount < 1    THEN '1. Under 1'
        WHEN amount < 10   THEN '2. 1-10'
        WHEN amount < 100  THEN '3. 10-100'
        WHEN amount < 500  THEN '4. 100-500'
        ELSE                    '5. 500+'
    END                                                         AS amount_band,
    COUNT(*)                                                    AS transactions,
    SUM(class)                                                  AS fraud_cases,
    ROUND(100.0 * SUM(class) / COUNT(*), 3)                     AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN class = 1 THEN amount ELSE 0 END), 0)   AS fraud_amount
FROM transactions
GROUP BY amount_band
ORDER BY amount_band;
