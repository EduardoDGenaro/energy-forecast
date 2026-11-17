"""Ingestao idempotente e incremental de dados brutos do ONS para a camada Bronze."""

from __future__ import annotations

import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import requests

from energy_forecast.utils.config import AppSettings
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)

S3_XML_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
PARQUET_FILENAME_PATTERN = re.compile(r"^CARGA_ENERGIA_(\d{4})\.parquet$")
EXPECTED_RAW_COLUMNS = (
    "id_subsistema",
    "nom_subsistema",
    "din_instante",
    "val_cargaenergiamwmed",
)


class IngestionError(RuntimeError):
    """Excecao lancada quando ocorre falha na descoberta, download ou validacao da Bronze."""


@dataclass(frozen=True)
class RemoteParquetObject:
    """Metadados de um arquivo anual Parquet disponivel no bucket S3 publico do ONS."""

    year: int
    file_name: str
    s3_key: str
    s3_uri: str
    source_url: str
    remote_size_bytes: int
    remote_etag: str
    remote_last_modified: str


@dataclass(frozen=True)
class ManifestFileEntry:
    """Registro auditavel de um arquivo persistido na camada Bronze."""

    year: int
    file_name: str
    source_url: str
    s3_uri: str
    remote_etag: str
    remote_last_modified: str
    size_bytes: int
    sha256: str
    row_count: int
    period_start: str
    period_end: str
    subsystems: list[str]
    physical_schema: dict[str, str]
    ingested_at_utc: str
    last_checked_at_utc: str
    last_action: str


@dataclass(frozen=True)
class IngestionSummary:
    """Resumo consolidado de uma execucao do pipeline de ingestao Bronze."""

    total_discovered: int
    downloaded_count: int
    skipped_count: int
    total_rows: int
    years_processed: list[int]
    manifest_path: Path


def compute_sha256_bytes(payload: bytes) -> str:
    """Calcula o hash SHA-256 hexadecimal de um buffer em memoria."""
    return hashlib.sha256(payload).hexdigest()


def compute_sha256_file(file_path: Path) -> str:
    """Calcula o hash SHA-256 hexadecimal de um arquivo em disco."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_s3_listing_xml(
    xml_text: str,
    s3_base_url: str,
    s3_uri_prefix: str,
    start_year: int,
    end_year: int | None = None,
) -> list[RemoteParquetObject]:
    """Extrai a lista de arquivos Parquet anuais a partir do XML ListBucketResult do S3.

    Args:
        xml_text: Conteudo XML retornado pela listagem S3 (list-type=2).
        s3_base_url: URL HTTPS base da pasta do dataset no S3.
        s3_uri_prefix: Prefixo URI S3 (ex: s3://ons-aws-prod-opendata/dataset/carga_energia_di/).
        start_year: Ano inicial minimo a considerar (ex: 2018).
        end_year: Ano final opcional para recorte superior.

    Returns:
        list[RemoteParquetObject]: Lista de objetos remotos ordenada cronologicamente por ano.

    Raises:
        IngestionError: Caso o XML seja invalido.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise IngestionError(f"Falha ao interpretar XML de listagem do S3: {exc}") from exc

    normalized_base_url = s3_base_url.rstrip("/")
    normalized_s3_prefix = s3_uri_prefix.rstrip("/")
    discovered: list[RemoteParquetObject] = []

    for contents in root.findall("s3:Contents", S3_XML_NAMESPACE):
        key_elem = contents.find("s3:Key", S3_XML_NAMESPACE)
        size_elem = contents.find("s3:Size", S3_XML_NAMESPACE)
        etag_elem = contents.find("s3:ETag", S3_XML_NAMESPACE)
        last_mod_elem = contents.find("s3:LastModified", S3_XML_NAMESPACE)

        if key_elem is None or not key_elem.text:
            continue

        s3_key = key_elem.text.strip()
        file_name = s3_key.rsplit("/", maxsplit=1)[-1]
        match = PARQUET_FILENAME_PATTERN.match(file_name)
        if not match:
            continue

        year = int(match.group(1))
        if year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue

        size_bytes = int(size_elem.text) if size_elem is not None and size_elem.text else 0
        raw_etag = (
            etag_elem.text.strip().strip('"') if etag_elem is not None and etag_elem.text else ""
        )
        last_mod = (
            last_mod_elem.text.strip() if last_mod_elem is not None and last_mod_elem.text else ""
        )

        discovered.append(
            RemoteParquetObject(
                year=year,
                file_name=file_name,
                s3_key=s3_key,
                s3_uri=f"{normalized_s3_prefix}/{file_name}",
                source_url=f"{normalized_base_url}/{file_name}",
                remote_size_bytes=size_bytes,
                remote_etag=raw_etag,
                remote_last_modified=last_mod,
            )
        )

    discovered.sort(key=lambda item: item.year)
    return discovered


def discover_remote_parquet_files(
    settings: AppSettings,
    start_year: int | None = None,
    end_year: int | None = None,
    session: requests.Session | None = None,
) -> list[RemoteParquetObject]:
    """Consulta o bucket S3 publico do ONS para listar os arquivos Parquet disponiveis.

    Args:
        settings: Configuracoes globais do projeto.
        start_year: Ano inicial opcional (padrao: settings.ingestion.start_year).
        end_year: Ano final opcional.
        session: Sessao HTTP opcional para reutilizacao de conexao ou testes.

    Returns:
        list[RemoteParquetObject]: Lista de arquivos encontrados a partir de start_year.

    Raises:
        IngestionError: Se a listagem falhar ou nenhum arquivo for encontrado.
    """
    effective_start_year = start_year if start_year is not None else settings.ingestion.start_year
    s3_base_url = settings.ingestion.s3_base_url.rstrip("/")

    # Extrai host do bucket e prefixo a partir de s3_base_url
    # Exemplo: https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di
    parts = s3_base_url.split("/", maxsplit=3)
    if len(parts) < 4:
        raise IngestionError(f"Formato inesperado para s3_base_url: {s3_base_url}")
    bucket_root_url = f"{parts[0]}//{parts[2]}"
    prefix = f"{parts[3].strip('/')}/"

    http = session or requests.Session()
    try:
        response = http.get(
            bucket_root_url,
            params={"list-type": "2", "prefix": prefix},
            timeout=settings.ingestion.request_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise IngestionError(
            f"Erro de rede ao listar arquivos no bucket S3 do ONS ({bucket_root_url}): {exc}"
        ) from exc

    files = parse_s3_listing_xml(
        xml_text=response.text,
        s3_base_url=s3_base_url,
        s3_uri_prefix=settings.ingestion.s3_uri_prefix,
        start_year=effective_start_year,
        end_year=end_year,
    )

    if not files:
        raise IngestionError(
            "Nenhum arquivo Parquet encontrado no bucket S3 para o "
            f"periodo >= {effective_start_year}."
        )

    return files


def inspect_raw_parquet_bytes(raw_bytes: bytes, file_name: str) -> dict[str, Any]:
    """Inspeciona os metadados de um arquivo Parquet em memoria sem modificar seu conteudo.

    Args:
        raw_bytes: Conteudo binario exato do arquivo Parquet baixado.
        file_name: Nome do arquivo para mensagens de diagnostico.

    Returns:
        dict[str, Any]: Metadados contendo row_count, period_start, period_end, subsystems e
            physical_schema.

    Raises:
        IngestionError: Caso o arquivo nao seja um Parquet valido ou nao contenha as colunas do ONS.
    """
    try:
        table = pq.read_table(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise IngestionError(
            f"Arquivo Parquet invalido ou corrompido ({file_name}): {exc}"
        ) from exc

    physical_schema = {field.name: str(field.type) for field in table.schema}
    missing_cols = [col for col in EXPECTED_RAW_COLUMNS if col not in physical_schema]
    if missing_cols:
        raise IngestionError(
            f"Colunas obrigatorias ausentes no arquivo {file_name}: {missing_cols}. "
            f"Colunas encontradas: {list(physical_schema.keys())}"
        )

    if table.num_rows == 0:
        raise IngestionError(f"O arquivo Parquet {file_name} nao contem registros (0 linhas).")

    df_meta = table.select(["id_subsistema", "din_instante"]).to_pandas()
    timestamps = pd.to_datetime(df_meta["din_instante"], errors="raise")
    period_start = timestamps.min().strftime("%Y-%m-%d")
    period_end = timestamps.max().strftime("%Y-%m-%d")
    subsystems = sorted(df_meta["id_subsistema"].dropna().astype(str).unique().tolist())

    return {
        "row_count": int(table.num_rows),
        "period_start": period_start,
        "period_end": period_end,
        "subsystems": subsystems,
        "physical_schema": physical_schema,
    }


def load_manifest(manifest_path: Path) -> dict[int, ManifestFileEntry]:
    """Carrega o manifesto de ingestao existente em disco, indexado por ano.

    Args:
        manifest_path: Caminho para ingestion_manifest.json.

    Returns:
        dict[int, ManifestFileEntry]: Dicionario mapeando ano -> entrada do manifesto.
    """
    if not manifest_path.is_file():
        return {}

    try:
        raw_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries: dict[int, ManifestFileEntry] = {}
        for item in raw_data.get("files", []):
            entry = ManifestFileEntry(
                year=int(item["year"]),
                file_name=str(item["file_name"]),
                source_url=str(item["source_url"]),
                s3_uri=str(item["s3_uri"]),
                remote_etag=str(item.get("remote_etag", "")),
                remote_last_modified=str(item.get("remote_last_modified", "")),
                size_bytes=int(item["size_bytes"]),
                sha256=str(item["sha256"]),
                row_count=int(item["row_count"]),
                period_start=str(item["period_start"]),
                period_end=str(item["period_end"]),
                subsystems=[str(s) for s in item.get("subsystems", [])],
                physical_schema={
                    str(k): str(v) for k, v in item.get("physical_schema", {}).items()
                },
                ingested_at_utc=str(item["ingested_at_utc"]),
                last_checked_at_utc=str(item.get("last_checked_at_utc", item["ingested_at_utc"])),
                last_action=str(item.get("last_action", "downloaded")),
            )
            entries[entry.year] = entry
        return entries
    except Exception as exc:
        logger.warning(
            "Nao foi possivel ler o manifesto existente em %s (%s). Um novo manifesto sera gerado.",
            manifest_path,
            exc,
        )
        return {}


def save_manifest(
    manifest_path: Path,
    settings: AppSettings,
    entries_by_year: dict[int, ManifestFileEntry],
) -> None:
    """Persiste o manifesto de ingestao em formato JSON estruturado e deterministico.

    Args:
        manifest_path: Caminho de destino do arquivo JSON.
        settings: Configuracoes da aplicacao.
        entries_by_year: Entradas auditadas por ano.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_entries = [asdict(entries_by_year[year]) for year in sorted(entries_by_year)]

    payload = {
        "dataset_slug": settings.ingestion.dataset_slug,
        "portal_url": "https://dados.ons.org.br/dataset/carga-energia",
        "ckan_package_url": settings.ingestion.ckan_package_url,
        "s3_uri_prefix": settings.ingestion.s3_uri_prefix,
        "target_subsystem": settings.ingestion.target_subsystem,
        "target_unit": settings.ingestion.target_unit,
        "start_year": settings.ingestion.start_year,
        "updated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_files": len(ordered_entries),
        "total_rows": sum(item["row_count"] for item in ordered_entries),
        "files": ordered_entries,
    }

    temp_path = manifest_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(manifest_path)


def is_local_file_up_to_date(
    local_path: Path,
    remote_obj: RemoteParquetObject,
    existing_entry: ManifestFileEntry | None,
) -> bool:
    """Verifica se o arquivo local na Bronze e identico a versao remota atual do ONS.

    Args:
        local_path: Caminho do arquivo Parquet local na camada Bronze.
        remote_obj: Metadados remotos retornados pelo S3 do ONS.
        existing_entry: Registro anterior no manifesto de ingestao, se existir.

    Returns:
        bool: True se o arquivo local ja existe, esta integro e nao sofreu atualizacao na origem.
    """
    if existing_entry is None or not local_path.is_file():
        return False

    local_size = local_path.stat().st_size
    if local_size != remote_obj.remote_size_bytes or local_size != existing_entry.size_bytes:
        return False

    if remote_obj.remote_etag and existing_entry.remote_etag != remote_obj.remote_etag:
        return False

    if (
        remote_obj.remote_last_modified
        and existing_entry.remote_last_modified != remote_obj.remote_last_modified
    ):
        return False

    local_sha256 = compute_sha256_file(local_path)
    return local_sha256 == existing_entry.sha256


def run_bronze_ingestion(
    settings: AppSettings,
    start_year: int | None = None,
    end_year: int | None = None,
    force: bool = False,
    session: requests.Session | None = None,
) -> IngestionSummary:
    """Executa a ingestao incremental e idempotente dos arquivos Parquet do ONS na camada Bronze.

    Preserva os bytes originais fornecidos pelo ONS sem qualquer transformacao de dados ou schema,
    validando a estrutura basica e registrando origem, periodo, schema fisico e hashes no manifesto.

    Args:
        settings: Configuracoes carregadas do projeto.
        start_year: Ano inicial opcional (padrao: 2018).
        end_year: Ano final opcional.
        force: Se True, rebaixa todos os arquivos mesmo que ja estejam atualizados localmente.
        session: Sessao HTTP opcional para requests.

    Returns:
        IngestionSummary: Resumo quantitativo da execucao.
    """
    bronze_dir = settings.paths.bronze_dir
    manifest_path = settings.paths.manifest_path
    bronze_dir.mkdir(parents=True, exist_ok=True)

    http = session or requests.Session()
    remote_files = discover_remote_parquet_files(
        settings=settings,
        start_year=start_year,
        end_year=end_year,
        session=http,
    )

    manifest_entries = load_manifest(manifest_path)
    downloaded_count = 0
    skipped_count = 0
    years_processed: list[int] = []

    for remote_obj in remote_files:
        local_file_path = bronze_dir / remote_obj.file_name
        existing_entry = manifest_entries.get(remote_obj.year)
        now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        if not force and is_local_file_up_to_date(local_file_path, remote_obj, existing_entry):
            assert existing_entry is not None
            manifest_entries[remote_obj.year] = ManifestFileEntry(
                year=existing_entry.year,
                file_name=existing_entry.file_name,
                source_url=existing_entry.source_url,
                s3_uri=existing_entry.s3_uri,
                remote_etag=existing_entry.remote_etag,
                remote_last_modified=existing_entry.remote_last_modified,
                size_bytes=existing_entry.size_bytes,
                sha256=existing_entry.sha256,
                row_count=existing_entry.row_count,
                period_start=existing_entry.period_start,
                period_end=existing_entry.period_end,
                subsystems=existing_entry.subsystems,
                physical_schema=existing_entry.physical_schema,
                ingested_at_utc=existing_entry.ingested_at_utc,
                last_checked_at_utc=now_utc,
                last_action="skipped_up_to_date",
            )
            skipped_count += 1
            years_processed.append(remote_obj.year)
            logger.info(
                "Ano %d ignorado (idempotente, arquivo integro e atualizado): %s",
                remote_obj.year,
                local_file_path.name,
            )
            continue

        logger.info(
            "Baixando arquivo bruto do ONS (ano %d): %s",
            remote_obj.year,
            remote_obj.source_url,
        )
        try:
            response = http.get(
                remote_obj.source_url,
                timeout=settings.ingestion.request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise IngestionError(f"Falha no download de {remote_obj.source_url}: {exc}") from exc

        raw_bytes = response.content
        metadata = inspect_raw_parquet_bytes(raw_bytes, remote_obj.file_name)
        sha256_hex = compute_sha256_bytes(raw_bytes)

        # Escrita atomica preservando os bytes originais sem qualquer transformacao
        temp_file = local_file_path.with_suffix(".parquet.tmp")
        temp_file.write_bytes(raw_bytes)
        temp_file.replace(local_file_path)

        manifest_entries[remote_obj.year] = ManifestFileEntry(
            year=remote_obj.year,
            file_name=remote_obj.file_name,
            source_url=remote_obj.source_url,
            s3_uri=remote_obj.s3_uri,
            remote_etag=remote_obj.remote_etag,
            remote_last_modified=remote_obj.remote_last_modified,
            size_bytes=len(raw_bytes),
            sha256=sha256_hex,
            row_count=metadata["row_count"],
            period_start=metadata["period_start"],
            period_end=metadata["period_end"],
            subsystems=metadata["subsystems"],
            physical_schema=metadata["physical_schema"],
            ingested_at_utc=now_utc,
            last_checked_at_utc=now_utc,
            last_action="downloaded",
        )
        downloaded_count += 1
        years_processed.append(remote_obj.year)
        logger.info(
            "Ano %d persistido na Bronze: arquivo=%s, linhas=%d, periodo=[%s .. %s], bytes=%d",
            remote_obj.year,
            local_file_path.name,
            metadata["row_count"],
            metadata["period_start"],
            metadata["period_end"],
            len(raw_bytes),
        )

    save_manifest(manifest_path, settings, manifest_entries)
    total_rows = sum(
        manifest_entries[yr].row_count for yr in years_processed if yr in manifest_entries
    )

    logger.info(
        "Ingestao Bronze concluida: descobertos=%d, baixados=%d, ignorados=%d, "
        "linhas_totais=%d, manifesto=%s",
        len(remote_files),
        downloaded_count,
        skipped_count,
        total_rows,
        manifest_path,
    )

    return IngestionSummary(
        total_discovered=len(remote_files),
        downloaded_count=downloaded_count,
        skipped_count=skipped_count,
        total_rows=total_rows,
        years_processed=years_processed,
        manifest_path=manifest_path,
    )
