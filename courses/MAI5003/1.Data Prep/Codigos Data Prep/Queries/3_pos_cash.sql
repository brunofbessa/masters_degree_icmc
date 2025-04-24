--tratando a pos cash

CREATE TABLE tabelas_manipuladas.pos_cash_balance_prep AS
SELECT 
    sk_id_curr, 
    sk_id_prev,
    AVG(cnt_installment) AS cnt_installment_pos_cash,
    AVG(cnt_installment_future) AS cnt_installment_future_pos_cash,
    MAX(sk_dpd) as sk_dpd_pos_cash,
    MAX(sk_dpd_def) as sk_dpd_def_pos_cash
    FROM 
        bases_kaggle.POS_CASH_balance
    WHERE 
        name_contract_status = 'Active'
    GROUP BY sk_id_curr, sk_id_prev; 
