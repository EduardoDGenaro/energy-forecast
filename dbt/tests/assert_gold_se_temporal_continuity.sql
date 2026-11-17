-- Teste de qualidade singular: garante ausencia de lacunas temporais (gaps > 1 dia) na serie Gold SE
select
    data_referencia,
    data_observacao_anterior,
    dias_desde_observacao_anterior
from {{ ref('fct_carga_diaria_se') }}
where dias_desde_observacao_anterior != 1
   or is_continuous = false
