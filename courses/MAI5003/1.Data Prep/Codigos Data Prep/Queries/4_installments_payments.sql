---Tratando installments payments
CREATE TABLE tabelas_manipuladas.installments_payments_prep AS
SELECT
    sk_id_curr,
    sk_id_prev,
    MAX(gap_days_payment_installments) AS gap_days_payment_installments,
    AVG(gap_amt_payment_installments) AS gap_amt_payment_installments
FROM
    (SELECT 
        sk_id_prev,
        sk_id_curr,
        (-1)*(days_entry_payment - days_instalment) as gap_days_payment_installments,
        (amt_payment - amt_instalment) as gap_amt_payment_installments
    FROM 
        bases_kaggle.installments_payments)
GROUP BY sk_id_curr, sk_id_prev;