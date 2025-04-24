
CREATE TABLE tabelas_manipuladas.previous_application_prep AS
SELECT 
A.*,		
B.cnt_historical_previous,	 		
B.amt_balance_previous,	 		
B.amt_credit_limit_previous,	 		
B.amt_drawings_atm_previous,	 		
B.amt_drawings_previous,	 		
B.amt_drawings_other_previous,	 		
B.amt_drawings_pos_previous,	 		
B.amt_inst_min_regularity_previous,	 		
B.amt_payment_previous,	 		
B.amt_payment_total_previous,	 		
B.amt_receivable_principal_previous,	 		
B.amt_recivable_previous,	 		
B.amt_total_receivable_previous,	 		
B.cnt_drawings_atm_previous,	 		
B.cnt_drawings_previous,	 		
B.cnt_drawings_other_previous,	 		
B.cnt_drawings_pos_previous,	 		
B.cnt_instalment_mature_cum_previous,	 		
B.sk_dpd_previous,	 		
B.sk_dpd_def_previous,	 		
(B.slope_amt_balance -      MIN(B.slope_amt_balance)      OVER ())/(MAX(B.slope_amt_balance)      OVER() - MIN(B.slope_amt_balance)      OVER ()) slope_amt_balance,	 		
(B.slope_amt_credit_limit - MIN(B.slope_amt_credit_limit) OVER ())/(MAX(B.slope_amt_credit_limit) OVER() - MIN(B.slope_amt_credit_limit) OVER ()) slope_amt_credit_limit,	 		
(B.slope_amt_payment -      MIN(B.slope_amt_payment)      OVER ())/(MAX(B.slope_amt_payment)      OVER() - MIN(B.slope_amt_payment)      OVER ()) slope_amt_payment,	 		
C.gap_days_payment_installments,
C.gap_amt_payment_installments,
D.cnt_installment_pos_cash,	
D.cnt_installment_future_pos_cash,	
D.sk_dpd_pos_cash,	
D.sk_dpd_def_pos_cash

FROM 
bases_kaggle.previous_application AS A 

LEFT JOIN 
tabelas_manipuladas.credit_card_balance_prep AS B 
ON A.sk_id_prev = b.sk_id_prev

LEFT JOIN 
tabelas_manipuladas.installments_payments_prep AS C 
ON A.sk_id_prev = b.sk_id_prev

LEFT JOIN
tabelas_manipuladas.pos_cash_prep AS D 
ON A.sk_id_prev = D.sk_id_prev;

