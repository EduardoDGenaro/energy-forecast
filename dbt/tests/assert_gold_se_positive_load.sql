-- Teste de qualidade singular: garante que todos os valores de carga na Gold SE sao estritamente positivos
select
    data_referencia,
    carga_mwmed
from {{ ref('fct_carga_diaria_se') }}
where carga_mwmed is null
   or carga_mwmed <= 0
   or not isfinite(carga_mwmed)
