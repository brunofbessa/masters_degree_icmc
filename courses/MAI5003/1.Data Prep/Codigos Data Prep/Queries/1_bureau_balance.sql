-- Vamos comecar a integracao do bureau 

CREATE TABLE bases_kaggle.bureau_balance_prep AS
SELECT A.sk_id_bureau, 
       A.delinquency_6_mths, 
       A.delinquency_6_12_mths, 
       A.delinquency_12_24_mths, 
       A.delinquency_24_36_mths, 
       A.delinquency_long,
       -1*(B.historical_size) AS historical_size
    FROM
        (SELECT
            sk_id_bureau, 
            MAX(CASE 
                WHEN months_cat = 'mths_6' THEN qtd_delinquency ELSE 0 END) AS delinquency_6_mths,
            MAX(CASE 
                WHEN months_cat = 'mths_6_12' THEN qtd_delinquency ELSE 0 END) AS delinquency_6_12_mths,
            MAX(CASE 
                WHEN months_cat = 'mths_12_24' THEN qtd_delinquency ELSE 0 END) AS delinquency_12_24_mths,
            MAX(CASE 
                WHEN months_cat = 'mths_24_36' THEN qtd_delinquency ELSE 0 END) AS delinquency_24_36_mths,
            MAX(CASE 
                WHEN months_cat = 'mths_36' THEN qtd_delinquency ELSE 0 END) AS delinquency_long
            FROM     
                (SELECT 
                    sk_id_bureau,
                    SUM(delinquency) as qtd_delinquency,
                    months_cat
                    FROM
                        (SELECT 
                            sk_id_bureau,
                            delinquency,
                            CASE 
                                WHEN months_balance between -6 AND 0 THEN 'mths_6'
                                WHEN months_balance between -12 AND -6 THEN 'mths_6_12'
                                WHEN months_balance between -24 AND -12 THEN 'mths_12_24'
                                WHEN months_balance between -36 AND -24 THEN 'mths_24_36'
                                WHEN months_balance < -36 THEN 'mths_36'
                                ELSE ''
                                END AS months_cat
                            FROM
                                (SELECT
                                    sk_id_bureau, 
                                    months_balance,
                                    CASE 
                                        WHEN STATUS IN ('X', 'C', '0') THEN 0 ELSE 1 END AS delinquency
                                    FROM bases_kaggle.bureau_balance))
                    GROUP BY sk_id_bureau, months_cat)
        GROUP BY sk_id_bureau) A        
    LEFT JOIN 
        (SELECT 
            sk_id_bureau,
            MIN(months_balance) AS historical_size
            FROM 
                bases_kaggle.bureau_balance
            GROUP BY sk_id_bureau) B
    ON A.sk_id_bureau = B.sk_id_bureau;

