


CREATE TABLE tabelas_manipuladas.credit_card_balance_prep AS
SELECT 
    A.sk_id_prev,
    COUNT(A.sk_id_prev) AS cnt_historical_previous,
    AVG(A.amt_balance) AS amt_balance_previous,
    AVG(A.amt_credit_limit_actual) AS amt_credit_limit_previous,
    AVG(A.amt_drawings_atm_current) AS amt_drawings_atm_previous,
    AVG(A.amt_drawings_current) AS amt_drawings_previous,
    AVG(A.amt_drawings_other_current) AS amt_drawings_other_previous,
    AVG(A.amt_drawings_pos_current) AS amt_drawings_pos_previous,
    AVG(A.amt_inst_min_regularity) AS amt_inst_min_regularity_previous,
    AVG(A.amt_payment_current) AS amt_payment_previous,
    AVG(A.amt_payment_total_current) AS amt_payment_total_previous,
    AVG(A.amt_receivable_principal) AS amt_receivable_principal_previous,
    AVG(A.amt_recivable) AS amt_recivable_previous,
    AVG(A.amt_total_receivable) AS amt_total_receivable_previous,
    AVG(A.cnt_drawings_atm_current) AS cnt_drawings_atm_previous,
    AVG(A.cnt_drawings_current) AS cnt_drawings_previous,
    AVG(A.cnt_drawings_other_current) AS cnt_drawings_other_previous,
    AVG(A.cnt_drawings_pos_current) AS cnt_drawings_pos_previous,
    AVG(A.cnt_instalment_mature_cum) AS cnt_instalment_mature_cum_previous,
    MAX(A.sk_dpd) AS sk_dpd_previous,
    MAX(A.sk_dpd_def) AS sk_dpd_def_previous,
    MAX(B.slope_amt_balance) AS slope_amt_balance,
    MAX(B.slope_amt_credit_limit) AS slope_amt_credit_limit,
    MAX(B.slope_amt_payment) AS slope_amt_payment
FROM
    bases_kaggle.credit_card_balance AS A 
LEFT JOIN 
    (SELECT 
        sk_id_prev,
        SUM(((-1)*months_balance - month_bar)*(amt_balance-amt_balance_bar))/SUM(((-1)*months_balance - month_bar)*((-1)*months_balance - month_bar)+1) AS slope_amt_balance,
        SUM(((-1)*months_balance - month_bar)*(amt_credit_limit_actual-amt_credit_limit_bar))/SUM(((-1)*months_balance - month_bar)*((-1)*months_balance - month_bar)+1) AS slope_amt_credit_limit,
        SUM(((-1)*months_balance - month_bar)*(amt_payment_current-amt_payment_bar))/SUM(((-1)*months_balance - month_bar)*((-1)*months_balance - month_bar)+1) AS slope_amt_payment,
    FROM    
        (SELECT  
            sk_id_prev,
            months_balance,
            amt_balance,
            amt_credit_limit_actual,
            amt_payment_current,
            (-1)*AVG(months_balance)             OVER (PARTITION BY sk_id_prev) AS month_bar,
            AVG(amt_balance)                     OVER (PARTITION BY sk_id_prev) AS amt_balance_bar,
            AVG(amt_credit_limit_actual)         OVER (PARTITION BY sk_id_prev) AS amt_credit_limit_bar,
            AVG(COALESCE(amt_payment_current,0)) OVER (PARTITION BY sk_id_prev) AS amt_payment_bar
        FROM bases_kaggle.credit_card_balance)
    GROUP BY sk_id_prev) AS B
ON A.sk_id_prev = B.sk_id_prev
WHERE A.name_contract_status = 'Active'
GROUP BY A.sk_id_prev;






AVG(name_contract_status) AS name_contract_status_previous,




months_balance













