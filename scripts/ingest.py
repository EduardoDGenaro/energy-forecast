"""Ponto de entrada CLI para ingestao de dados brutos do ONS na camada Bronze."""

from __future__ import annotations

import argparse
import sys

from energy_forecast.ingestion.bronze import IngestionError, run_bronze_ingestion
from energy_forecast.utils.config import ConfigurationError, load_settings
from energy_forecast.utils.logging import configure_logging, get_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta argumentos de linha de comando para a ingestao Bronze."""
    parser = argparse.ArgumentParser(
        description="Ingestao idempotente de dados publicos de carga diaria do ONS (camada Bronze)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Caminho opcional para o arquivo settings.yaml.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Ano inicial para ingestao (padrao definido em settings.yaml: 2018).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Ano final opcional para limitar o recorte superior da ingestao.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forca novo download mesmo quando os arquivos locais ja estao atualizados.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Executa o fluxo de ingestao Bronze e registra o manifesto de auditoria."""
    configure_logging()
    logger = get_logger(__name__)
    args = parse_args(argv)

    try:
        settings = load_settings(config_path=args.config)
        summary = run_bronze_ingestion(
            settings=settings,
            start_year=args.start_year,
            end_year=args.end_year,
            force=args.force,
        )
    except (ConfigurationError, IngestionError) as exc:
        logger.error("Falha na execucao da ingestao Bronze: %s", exc)
        return 1

    logger.info(
        "Resumo da ingestao Bronze: anos=%s, baixados=%d, ignorados=%d, linhas=%d",
        summary.years_processed,
        summary.downloaded_count,
        summary.skipped_count,
        summary.total_rows,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
