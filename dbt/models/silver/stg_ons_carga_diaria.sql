{{
    config(
        materialized='table',
        post_hook=[
            "COPY {{ this }} TO '../data/silver/stg_ons_carga_diaria.parquet' (FORMAT PARQUET)"
        ]
    )
}}

with raw_bronze as (
    select
        id_subsistema,
        nom_subsistema,
        din_instante,
        val_cargaenergiamwmed,
        filename
    from {{ source('bronze_ons', 'carga_energia_diaria') }}
),

standardized as (
    select
        upper(trim(cast(id_subsistema as varchar))) as id_subsistema,
        trim(cast(nom_subsistema as varchar)) as nom_subsistema,
        cast(din_instante as date) as data_referencia,
        cast(din_instante as timestamp) as din_instante_raw,
        cast(val_cargaenergiamwmed as varchar) as val_cargaenergiamwmed_raw,
        try_cast(val_cargaenergiamwmed as double) as val_cargaenergiamwmed,
        regexp_extract(replace(filename, '\\', '/'), '[^/]+$') as _bronze_source_file,
        replace(filename, '\\', '/') as _bronze_filepath
    from raw_bronze
),

audited as (
    select
        id_subsistema,
        nom_subsistema,
        data_referencia,
        din_instante_raw,
        val_cargaenergiamwmed_raw,
        val_cargaenergiamwmed,
        (
            id_subsistema in ('SE', 'S', 'NE', 'N')
            and nom_subsistema is not null
            and length(nom_subsistema) > 0
        ) as is_valid_subsystem,
        (
            data_referencia is not null
            and data_referencia >= cast('{{ var("start_date", "2018-01-01") }}' as date)
        ) as is_valid_date,
        (
            val_cargaenergiamwmed is not null
            and isfinite(val_cargaenergiamwmed)
            and val_cargaenergiamwmed > 0.0
        ) as is_valid_load,
        row_number() over (
            partition by id_subsistema, data_referencia
            order by _bronze_source_file desc
        ) as duplicate_rank,
        count(*) over (
            partition by id_subsistema, data_referencia
        ) as key_occurrence_count,
        _bronze_source_file,
        _bronze_filepath,
        current_timestamp as _processed_at_utc
    from standardized
)

select
    id_subsistema,
    nom_subsistema,
    data_referencia,
    din_instante_raw,
    val_cargaenergiamwmed_raw,
    val_cargaenergiamwmed,
    is_valid_subsystem,
    is_valid_date,
    is_valid_load,
    duplicate_rank,
    key_occurrence_count,
    (duplicate_rank > 1) as is_duplicate,
    (
        is_valid_subsystem
        and is_valid_date
        and is_valid_load
        and duplicate_rank = 1
    ) as is_valid_record,
    _bronze_source_file,
    _bronze_filepath,
    _processed_at_utc
from audited
order by data_referencia asc, id_subsistema asc
