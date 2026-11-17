"""Testes de integracao validando os contratos das camadas Silver e Gold geradas pelo dbt."""

from __future__ import annotations

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from energy_forecast.utils.config import load_settings


def test_silver_layer_contract_and_traceability() -> None:
    """Verifica que a camada Silver preserva todas as linhas da Bronze, tipagem e linhagem."""
    settings = load_settings()
    silver_parquet = settings.paths.silver_dir / "stg_ons_carga_diaria.parquet"
    assert silver_parquet.is_file(), f"Arquivo Parquet Silver nao encontrado: {silver_parquet}"

    table = pq.read_table(silver_parquet)
    df = table.to_pandas()

    # Nenhuma linha eliminada silenciosamente em relacao aos 9 arquivos da Bronze
    assert len(df) == 12960

    required_cols = {
        "id_subsistema",
        "nom_subsistema",
        "data_referencia",
        "val_cargaenergiamwmed_raw",
        "val_cargaenergiamwmed",
        "is_valid_subsystem",
        "is_valid_date",
        "is_valid_load",
        "duplicate_rank",
        "is_duplicate",
        "is_valid_record",
        "_bronze_source_file",
    }
    assert required_cols.issubset(set(df.columns))

    assert df["val_cargaenergiamwmed"].isna().sum() == 0
    assert (df["val_cargaenergiamwmed"] > 0).all()
    assert df["is_valid_record"].all()
    assert not df["is_duplicate"].any()
    assert set(df["id_subsistema"].unique()) == {"N", "NE", "S", "SE"}
    assert df["_bronze_source_file"].str.startswith("CARGA_ENERGIA_").all()


def test_gold_layer_se_time_series_continuity_and_ordering() -> None:
    """Verifica unicidade, ordenacao cronologica e continuidade diaria da serie Gold SE."""
    settings = load_settings()
    gold_parquet = settings.paths.gold_dir / "fct_carga_diaria_se.parquet"
    assert gold_parquet.is_file(), f"Arquivo Parquet Gold nao encontrado: {gold_parquet}"

    df = pq.read_table(gold_parquet).to_pandas()

    assert len(df) == 3240
    assert set(df["id_subsistema"].unique()) == {"SE"}
    assert set(df["nom_subsistema"].unique()) == {"Sudeste/Centro-Oeste"}

    dates = pd.to_datetime(df["data_referencia"])
    assert dates.is_monotonic_increasing
    assert dates.is_unique
    assert dates.min() == pd.Timestamp("2018-01-01")
    assert dates.max() == pd.Timestamp("2026-11-14")

    # Continuidade estrita: diferenca de exatamente 1 dia em toda a serie
    assert (df["dias_desde_observacao_anterior"] == 1).all()
    assert df["is_continuous"].all()
    assert (df["carga_mwmed"] > 0).all()


def test_duckdb_database_tables_match_parquet_exports() -> None:
    """Garante que as tabelas fisicas no arquivo DuckDB correspondem aos Parquets exportados."""
    settings = load_settings()
    assert settings.paths.duckdb_path.is_file()

    with duckdb.connect(str(settings.paths.duckdb_path), read_only=True) as con:
        silver_count = con.execute(
            "SELECT count(*) FROM main_silver.stg_ons_carga_diaria"
        ).fetchone()[0]
        gold_count = con.execute("SELECT count(*) FROM main_gold.fct_carga_diaria_se").fetchone()[0]

    assert silver_count == 12960
    assert gold_count == 3240
