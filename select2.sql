SELECT linha, 
"15_Train__01_1_06",
"15_Train__02_1_04",
"15_Val__01_1_06",
"15_Val__02_1_04",
"25_Train__01_1_03",
"25_Train__02_0_99",
"25_Val__01_1_03",
"25_Val__02_0_99",
"50_Train__01_1_00",
"50_Train__02_0_99",
"50_Val__01_1_00",
"50_Val__02_0_99"
 FROM crosstab(
  $$
    SELECT ARRAY[linha]::text[], linha,  coguinitive, result
    FROM processamento t 
    ORDER BY 1
  $$,
  $$
    SELECT distinct  coguinitive FROM processamento t2 ORDER BY 1
  $$
) AS c(rn text[], 
linha int,
"15_Train__01_1_06" int,
"15_Train__02_1_04" int,
"15_Val__01_1_06" int,
"15_Val__02_1_04" int,
"25_Train__01_1_03" int,
"25_Train__02_0_99" int,
"25_Val__01_1_03" int,
"25_Val__02_0_99" int,
"50_Train__01_1_00" int,
"50_Train__02_0_99" int,
"50_Val__01_1_00" int,
"50_Val__02_0_99" int )
--where ResultPredication_Train_weights_improvement_02_0_99 = '1'
 where "25_Train__02_0_99" = 1
 and "50_Train__02_0_99" = 0
order by linha
