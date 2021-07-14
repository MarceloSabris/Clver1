SELECT linha, qtd,
ResultPredication_Train_weights_improvement_01_1_00  ,
ResultPredication_Train_weights_improvement_01_1_03 , 
ResultPredication_Train_weights_improvement_01_1_06 ,
ResultPredication_Train_weights-improvement_02_0_99 , 
ResultPredication_Train_weights_improvement_02_0_99, 
ResultPredication_Train_weights_improvement_02_1_04 ,
ResultPredication_Val_weights_improvement_01_1_00 ,
ResultPredication_Val_weights-improvement_01_1_03  ,
ResultPredication_Val_weights_improvement_01_1_06 ,
ResultPredication_Val_weights_improvement_02_0_99  ,
ResultPredication_Val_weights_improvement_02_1_04  
FROM crosstab(
  $$
    SELECT ARRAY[linha,qtd]::text[], linha, qtd, coguinitive, result
    FROM processamento t 
    ORDER BY 1
  $$,
  $$
    SELECT distinct  coguinitive FROM processamento t2 ORDER BY 1
  $$
) AS c(rn text[], linha int,qtd text,
ResultPredication_Train_weights_improvement_01_1_00 int ,
ResultPredication_Train_weights_improvement_01_1_03 int , 
ResultPredication_Train_weights_improvement_01_1_06 int,
ResultPredication_Train_weights_improvement_02_0_99 int,
ResultPredication_Train_weights_improvement_02_1_04 int , 
ResultPredication_Val_weights_improvement_01_1_00 int, 
ResultPredication_Val_weights_improvement_01_1_03 int ,
ResultPredication_Val_weights_improvement_01_1_06 int,
ResultPredication_Val_weights_improvement_02_0_99 int , 
ResultPredication_Val_weights_improvement_02_1_04 int)
order by linha;

INSERT INTO processamento 
(linha, qtd, coguinitive, result)
VALUES(0, '25', 'teste10', '0');





select * from  processamento p where result ='1' 

commit transaction 


update processamento set coguinitive ='ResultPredication_Train_weights_improvement_02_0_99' 
where coguinitive = 'ResultPredication_Train_weights-improvement_02_0_99'
delete from mestrado.processamento
where coguinitive = 'ResultPredication_Train_weights_improvement_01_1_00'


-- mestrado.processamento definition

-- Drop table

-- DROP TABLE mestrado.processamento;

CREATE TABLE mestrado.processamento (
	linha int4 NOT NULL,
	qtd int4 NOT NULL,
	coguinitive text NOT NULL,
	"result" bit(1) NULL,
	CONSTRAINT processamento_pk PRIMARY KEY (linha, qtd, coguinitive)
);

select *  from processamento where coguinitive = 'ResultPredication_Train_weights_improvement_01_1_06'