---Tratando o Bureau
CREATE TABLE tabelas_manipuladas.bureau_prep AS
SELECT
    sk_id_curr,
    COUNT(DISTINCT sk_id_bureau) AS cnt_contracts_bureau,
    SUM(CASE WHEN credit_type = 'Consumer credit' THEN amt_credit_sum ELSE 0 END) AS amt_credit_sum_consumer_credit,
    SUM(CASE WHEN credit_type = 'Consumer credit' THEN amt_credit_sum_debt ELSE 0 END) AS amt_credit_sum_debt_consumer_credit,
    SUM(CASE WHEN credit_type = 'Consumer credit' THEN amt_credit_sum_limit ELSE 0 END) AS amt_credit_sum_limit_consumer_credit,
    SUM(CASE WHEN credit_type = 'Consumer credit' THEN amt_credit_sum_overdue ELSE 0 END) AS amt_credit_sum_overdue_consumer_credit,

    SUM(CASE WHEN credit_type = 'Credit card' THEN amt_credit_sum ELSE 0 END) AS amt_credit_sum_credit_card,
    SUM(CASE WHEN credit_type = 'Credit card' THEN amt_credit_sum_debt ELSE 0 END) AS amt_credit_sum_debt_credit_card,
    SUM(CASE WHEN credit_type = 'Credit card' THEN amt_credit_sum_limit ELSE 0 END) AS amt_credit_sum_limit_credit_card,
    SUM(CASE WHEN credit_type = 'Credit card' THEN amt_credit_sum_overdue ELSE 0 END) AS amt_credit_sum_overdue_credit_card,

    SUM(CASE WHEN credit_type = 'Loan for working capital replenishment' THEN amt_credit_sum ELSE 0 END) AS amt_credit_sum_working_capital,
    SUM(CASE WHEN credit_type = 'Loan for working capital replenishment' THEN amt_credit_sum_debt ELSE 0 END) AS amt_credit_sum_debt_working_capital,
    SUM(CASE WHEN credit_type = 'Loan for working capital replenishment' THEN amt_credit_sum_limit ELSE 0 END) AS amt_credit_sum_limit_working_capital,
    SUM(CASE WHEN credit_type = 'Loan for working capital replenishment' THEN amt_credit_sum_overdue ELSE 0 END) AS amt_credit_sum_overdue_working_capital,

    SUM(CASE WHEN credit_type = 'Microloan' THEN amt_credit_sum ELSE 0 END) AS amt_credit_sum_microloan,
    SUM(CASE WHEN credit_type = 'Microloan' THEN amt_credit_sum_debt ELSE 0 END) AS amt_credit_sum_debt_microloan,
    SUM(CASE WHEN credit_type = 'Microloan' THEN amt_credit_sum_limit ELSE 0 END) AS amt_credit_sum_limit_microloan,
    SUM(CASE WHEN credit_type = 'Microloan' THEN amt_credit_sum_overdue ELSE 0 END) AS amt_credit_sum_overdue_microloan,

    (-1)*MAX(days_credit) AS days_credit,
    MAX(credit_day_overdue) AS credit_day_overdue,
    MAX(days_credit_enddate) AS days_credit_enddate,
    MAX(amt_credit_max_overdue) AS amt_credit_max_overdue,
    MAX(cnt_credit_prolong) AS cnt_credit_prolong,
    MAX(days_credit_update) AS days_credit_update,
    MAX(historical_size) AS historical_size,

    SUM(CASE WHEN credit_type = 'Consumer credit' THEN delinquency_6_mths ELSE 0 END) AS delinquency_6_mths_consumer_credit,
    SUM(CASE WHEN credit_type = 'Consumer credit' THEN delinquency_6_12_mths ELSE 0 END) AS delinquency_6_12_mths_consumer_credit,
    SUM(CASE WHEN credit_type = 'Consumer credit' THEN delinquency_12_24_mths ELSE 0 END) AS delinquency_12_24_mths_consumer_credit,
    SUM(CASE WHEN credit_type = 'Consumer credit' THEN delinquency_24_36_mths ELSE 0 END) AS delinquency_24_36_mths_consumer_credit,
    SUM(CASE WHEN credit_type = 'Consumer credit' THEN delinquency_long ELSE 0 END) AS delinquency_long_consumer_credit,

    SUM(CASE WHEN credit_type = 'Credit card' THEN delinquency_6_mths ELSE 0 END) AS delinquency_6_mths_credit_card,
    SUM(CASE WHEN credit_type = 'Credit card' THEN delinquency_6_12_mths ELSE 0 END) AS delinquency_6_12_mths_credit_card,
    SUM(CASE WHEN credit_type = 'Credit card' THEN delinquency_12_24_mths ELSE 0 END) AS delinquency_12_24_mths_credit_card,
    SUM(CASE WHEN credit_type = 'Credit card' THEN delinquency_24_36_mths ELSE 0 END) AS delinquency_24_36_mths_credit_card,
    SUM(CASE WHEN credit_type = 'Credit card' THEN delinquency_long ELSE 0 END) AS delinquency_long_credit_card,

    SUM(CASE WHEN credit_type = 'Loan for working capital replenishment' THEN delinquency_6_mths ELSE 0 END) AS delinquency_6_mths_working_capital,
    SUM(CASE WHEN credit_type = 'Loan for working capital replenishment' THEN delinquency_6_12_mths ELSE 0 END) AS delinquency_6_12_mths_working_capital,
    SUM(CASE WHEN credit_type = 'Loan for working capital replenishment' THEN delinquency_12_24_mths ELSE 0 END) AS delinquency_12_24_mths_working_capital,
    SUM(CASE WHEN credit_type = 'Loan for working capital replenishment' THEN delinquency_24_36_mths ELSE 0 END) AS delinquency_24_36_mths_working_capital,
    SUM(CASE WHEN credit_type = 'Loan for working capital replenishment' THEN delinquency_long ELSE 0 END) AS delinquency_long_working_capital,

    SUM(CASE WHEN credit_type = 'Microloan' THEN delinquency_6_mths ELSE 0 END) AS delinquency_6_mths_microloan,
    SUM(CASE WHEN credit_type = 'Microloan' THEN delinquency_6_12_mths ELSE 0 END) AS delinquency_6_12_mths_microloan,
    SUM(CASE WHEN credit_type = 'Microloan' THEN delinquency_12_24_mths ELSE 0 END) AS delinquency_12_24_mths_microloan,
    SUM(CASE WHEN credit_type = 'Microloan' THEN delinquency_24_36_mths ELSE 0 END) AS delinquency_24_36_mths_microloan,
    SUM(CASE WHEN credit_type = 'Microloan' THEN delinquency_long ELSE 0 END) AS delinquency_long_microloan,
FROM
    (SELECT 
        A.*, 
        B.delinquency_6_mths,
        B.delinquency_6_12_mths,
        B.delinquency_12_24_mths,
        B.delinquency_24_36_mths,
        B.delinquency_long,
        B.historical_size
    FROM
        bases_kaggle.bureau AS A 
    LEFT JOIN 
        tabelas_manipuladas.bureau_balance_prep AS B 
    ON A.sk_id_bureau = B.sk_id_bureau
    WHERE TRIM(A.credit_active) = 'Active'
    AND TRIM(A.credit_currency) = 'currency 1')
GROUP BY sk_id_curr;
