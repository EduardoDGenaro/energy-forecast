{{
    config(
        materialized='table',
        post_hook=[
            "COPY {{ this }} TO '../data/gold/fct_carga_diaria_se.parquet' (FORMAT PARQUET)"
        ]
    )
}}

with silver_se as (
    select
        data_referencia,
        id_subsistema,
        nom_subsistema,
        val_cargaenergiamwmed,
        _bronze_source_file,
        _processed_at_utc
    from {{ ref('stg_ons_carga_diaria') }}
    where id_subsistema = '{{ var("target_subsystem", "SE") }}'
      and is_valid_record = true
),

ordered_series as (
    select
        data_referencia,
        id_subsistema,
        nom_subsistema,
        val_cargaenergiamwmed as carga_mwmed,
        val_cargaenergiamwmed,
        extract('year' from data_referencia) as ano,
        extract('month' from data_referencia) as mes,
        extract('day' from data_referencia) as dia,
        extract('isodow' from data_referencia) as dia_semana_iso,
        lag(data_referencia) over (order by data_referencia asc) as data_observacao_anterior,
        coalesce(
            date_diff(
                'day',
                lag(data_referencia) over (order by data_referencia asc),
                data_referencia
            ),
            1
        ) as dias_desde_observacao_anterior,
        _bronze_source_file,
        _processed_at_utc
    from silver_se
)

select
    data_referencia,
    id_subsistema,
    nom_subsistema,
    carga_mwmed,
    val_cargaenergiamwmed,
    ano,
    mes,
    dia,
    dia_semana_iso,
    data_observacao_anterior,
    dias_desde_observacao_anterior,
    (dias_desde_observacao_anterior = 1) as is_continuous,
    _bronze_source_file,
    _processed_at_utc
from ordered_series
order by data_referencia asc
