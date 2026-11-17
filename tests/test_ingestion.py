"""Testes automatizados para descoberta, validacao, idempotencia e manifesto da camada Bronze."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from energy_forecast.ingestion.bronze import (
    IngestionError,
    inspect_raw_parquet_bytes,
    load_manifest,
    parse_s3_listing_xml,
    run_bronze_ingestion,
)
from energy_forecast.utils.config import load_settings

SAMPLE_S3_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>ons-aws-prod-opendata</Name>
  <Prefix>dataset/carga_energia_di/</Prefix>
  <Contents>
    <Key>dataset/carga_energia_di/CARGA_ENERGIA_2017.parquet</Key>
    <LastModified>2024-08-26T18:27:02.000Z</LastModified>
    <ETag>"etag-2017"</ETag>
    <Size>24307</Size>
  </Contents>
  <Contents>
    <Key>dataset/carga_energia_di/CARGA_ENERGIA_2018.parquet</Key>
    <LastModified>2024-08-26T18:27:25.000Z</LastModified>
    <ETag>"etag-2018"</ETag>
    <Size>100</Size>
  </Contents>
  <Contents>
    <Key>dataset/carga_energia_di/CARGA_ENERGIA_2022.parquet</Key>
    <LastModified>2025-07-28T15:33:15.000Z</LastModified>
    <ETag>"etag-2022"</ETag>
    <Size>200</Size>
  </Contents>
  <Contents>
    <Key>dataset/carga_energia_di/DicionarioDados_Carga_Energia_Diaria.pdf</Key>
    <LastModified>2024-10-15T18:17:48.000Z</LastModified>
    <ETag>"etag-pdf"</ETag>
    <Size>167198</Size>
  </Contents>
</ListBucketResult>
"""


def _build_parquet_bytes(year: int, load_as_string: bool = False) -> bytes:
    """Constroi um arquivo Parquet em memoria reproduzindo o schema real do ONS."""
    df = pd.DataFrame(
        {
            "id_subsistema": ["SE", "S", "NE", "N"],
            "nom_subsistema": [
                "Sudeste/Centro-Oeste",
                "Sul",
                "Nordeste",
                "Norte",
            ],
            "din_instante": pd.to_datetime(
                [
                    f"{year}-01-01",
                    f"{year}-01-01",
                    f"{year}-01-02",
                    f"{year}-12-31",
                ]
            ),
            "val_cargaenergiamwmed": (
                ["38450.12000000", "11200.50000000", "9800.25000000", "5400.10000000"]
                if load_as_string
                else [38450.12, 11200.50, 9800.25, 5400.10]
            ),
        }
    )
    table = pa.Table.from_pandas(df, preserve_index=False)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


class _FakeResponse:
    """Resposta HTTP simulada para testes isolados sem rede."""

    def __init__(self, text: str = "", content: bytes = b"") -> None:
        self.text = text
        self.content = content

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    """Sessao HTTP simulada para testar ingestao, idempotencia e manifesto."""

    def __init__(self, xml_listing: str, payloads_by_url: dict[str, bytes]) -> None:
        self.xml_listing = xml_listing
        self.payloads_by_url = payloads_by_url
        self.downloaded_urls: list[str] = []

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        timeout: int = 60,
    ) -> _FakeResponse:
        if params and params.get("list-type") == "2":
            return _FakeResponse(text=self.xml_listing)
        self.downloaded_urls.append(url)
        return _FakeResponse(content=self.payloads_by_url[url])


def test_parse_s3_listing_filters_years_and_non_parquet() -> None:
    """Garante que apenas arquivos CARGA_ENERGIA_YYYY.parquet >= start_year sao selecionados."""
    objects = parse_s3_listing_xml(
        xml_text=SAMPLE_S3_XML,
        s3_base_url="https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di",
        s3_uri_prefix="s3://ons-aws-prod-opendata/dataset/carga_energia_di/",
        start_year=2018,
    )

    assert [obj.year for obj in objects] == [2018, 2022]
    assert objects[0].file_name == "CARGA_ENERGIA_2018.parquet"
    assert objects[0].remote_etag == "etag-2018"
    assert objects[1].file_name == "CARGA_ENERGIA_2022.parquet"


def test_inspect_raw_parquet_supports_both_2018_string_and_2022_double_schemas() -> None:
    """Verifica a inspecao sem mutacao para os schemas string (2018-2021) e double (2022+)."""
    bytes_2018 = _build_parquet_bytes(2018, load_as_string=True)
    meta_2018 = inspect_raw_parquet_bytes(bytes_2018, "CARGA_ENERGIA_2018.parquet")

    assert meta_2018["row_count"] == 4
    assert meta_2018["period_start"] == "2018-01-01"
    assert meta_2018["period_end"] == "2018-12-31"
    assert meta_2018["subsystems"] == ["N", "NE", "S", "SE"]
    assert "string" in meta_2018["physical_schema"]["val_cargaenergiamwmed"]

    bytes_2022 = _build_parquet_bytes(2022, load_as_string=False)
    meta_2022 = inspect_raw_parquet_bytes(bytes_2022, "CARGA_ENERGIA_2022.parquet")
    assert meta_2022["physical_schema"]["val_cargaenergiamwmed"] == "double"


def test_inspect_raw_parquet_rejects_missing_columns() -> None:
    """Garante erro explicito se uma coluna obrigatoria do ONS estiver ausente."""
    bad_df = pd.DataFrame({"id_subsistema": ["SE"], "din_instante": [pd.Timestamp("2020-01-01")]})
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(bad_df, preserve_index=False), buf)

    with pytest.raises(IngestionError, match="Colunas obrigatorias ausentes"):
        inspect_raw_parquet_bytes(buf.getvalue(), "CARGA_ENERGIA_2020.parquet")


def test_idempotent_bronze_ingestion_and_manifest_generation(tmp_path: Path) -> None:
    """Valida download inicial, preservacao exata dos bytes, manifesto e reexecucao idempotente."""
    base_settings = load_settings()
    bronze_dir = tmp_path / "bronze"
    manifest_path = bronze_dir / "ingestion_manifest.json"
    custom_paths = replace(
        base_settings.paths,
        bronze_dir=bronze_dir,
        manifest_path=manifest_path,
    )
    settings = replace(base_settings, paths=custom_paths)

    bytes_2018 = _build_parquet_bytes(2018, load_as_string=True)
    bytes_2022 = _build_parquet_bytes(2022, load_as_string=False)

    xml_dynamic = f"""<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <Contents>
        <Key>dataset/carga_energia_di/CARGA_ENERGIA_2018.parquet</Key>
        <LastModified>2024-08-26T18:27:25.000Z</LastModified>
        <ETag>"etag-2018"</ETag>
        <Size>{len(bytes_2018)}</Size>
      </Contents>
      <Contents>
        <Key>dataset/carga_energia_di/CARGA_ENERGIA_2022.parquet</Key>
        <LastModified>2025-07-28T15:33:15.000Z</LastModified>
        <ETag>"etag-2022"</ETag>
        <Size>{len(bytes_2022)}</Size>
      </Contents>
    </ListBucketResult>
    """

    base_url = settings.ingestion.s3_base_url.rstrip("/")
    payloads = {
        f"{base_url}/CARGA_ENERGIA_2018.parquet": bytes_2018,
        f"{base_url}/CARGA_ENERGIA_2022.parquet": bytes_2022,
    }
    fake_session = _FakeSession(xml_listing=xml_dynamic, payloads_by_url=payloads)

    # 1a execucao: baixa ambos os arquivos
    summary_1 = run_bronze_ingestion(settings=settings, session=fake_session)
    assert summary_1.downloaded_count == 2
    assert summary_1.skipped_count == 0
    assert summary_1.total_rows == 8
    assert (bronze_dir / "CARGA_ENERGIA_2018.parquet").read_bytes() == bytes_2018
    assert (bronze_dir / "CARGA_ENERGIA_2022.parquet").read_bytes() == bytes_2022

    manifest_entries = load_manifest(manifest_path)
    assert set(manifest_entries.keys()) == {2018, 2022}
    assert manifest_entries[2018].last_action == "downloaded"

    # 2a execucao: nenhum arquivo mudou na origem -> 0 downloads, 2 ignorados
    fake_session_2 = _FakeSession(xml_listing=xml_dynamic, payloads_by_url=payloads)
    summary_2 = run_bronze_ingestion(settings=settings, session=fake_session_2)
    assert summary_2.downloaded_count == 0
    assert summary_2.skipped_count == 2
    assert len(fake_session_2.downloaded_urls) == 0

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw_manifest["total_files"] == 2
    assert raw_manifest["files"][0]["last_action"] == "skipped_up_to_date"
