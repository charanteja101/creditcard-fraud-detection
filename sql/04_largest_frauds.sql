-- The 20 largest fraudulent transactions.
SELECT transaction_id, ROUND(time_seconds, 0) AS time_seconds, amount
FROM transactions
WHERE class = 1
ORDER BY amount DESC
LIMIT 20;
