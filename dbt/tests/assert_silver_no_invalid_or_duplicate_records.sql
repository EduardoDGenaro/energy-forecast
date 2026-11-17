-- Teste de qualidade singular: sinaliza caso existam registros invalidos ou duplicados na camada Silver
select
    id_subsistema,
    data_referencia,
    val_cargaenergiamwmed_raw,
    is_valid_subsystem,
    is_valid_date,
    is_valid_load,
    is_duplicate,
    _bronze_source_file
from {{ ref('stg_ons_carga_diaria') }}
where is_valid_record = false
