"""Ponto de entrada CLI para execucao de comandos dbt sobre o banco DuckDB."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from energy_forecast.utils.config import load_settings
from energy_forecast.utils.logging import configure_logging, get_logger


def _resolve_dbt_executable() -> str:
    """Localiza o executavel do dbt no mesmo ambiente do interpretador Python atual."""
    venv_bin_dir = Path(sys.executable).resolve().parent
    for candidate_name in ("dbt.exe", "dbt"):
        candidate_path = venv_bin_dir / candidate_name
        if candidate_path.is_file():
            return str(candidate_path)
    which_dbt = shutil.which("dbt")
    return which_dbt if which_dbt is not None else "dbt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta argumentos de linha de comando para execucao do dbt."""
    parser = argparse.ArgumentParser(
        description="Executa transformacoes e testes dbt (camadas Silver e Gold) no DuckDB."
    )
    parser.add_argument(
        "--command",
        type=str,
        choices=["debug", "compile", "run", "test", "build"],
        default="build",
        help="Comando dbt a ser executado (padrao: build).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="dev",
        help="Target do profile dbt (padrao: dev).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Caminho opcional para o arquivo settings.yaml.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Executa o comando dbt configurado apontando para o projeto e banco DuckDB."""
    configure_logging()
    logger = get_logger(__name__)
    args = parse_args(argv)

    settings = load_settings(config_path=args.config)
    dbt_dir = settings.paths.dbt_project_dir
    profiles_file = dbt_dir / "profiles.yml"
    example_profiles_file = dbt_dir / "profiles.yml.example"

    if not profiles_file.exists() and example_profiles_file.exists():
        profiles_file.write_text(
            example_profiles_file.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        logger.info("Arquivo dbt/profiles.yml inicializado a partir de profiles.yml.example.")

    env = os.environ.copy()
    env["DUCKDB_PATH"] = str(settings.paths.duckdb_path)
    env["DBT_PROFILES_DIR"] = str(dbt_dir)

    dbt_bin = _resolve_dbt_executable()
    cmd = [
        dbt_bin,
        args.command,
        "--project-dir",
        str(dbt_dir),
        "--profiles-dir",
        str(dbt_dir),
        "--target",
        args.target,
    ]
    logger.info("Executando comando dbt: %s", " ".join(cmd))

    completed = subprocess.run(cmd, env=env, cwd=str(dbt_dir), check=False)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
