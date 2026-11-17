"""Configuracao padronizada de logging estruturado para o projeto."""

from __future__ import annotations

import logging
import os
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(level: str | None = None) -> None:
    """Configura o logger raiz da aplicacao com formato estruturado.

    Args:
        level: Nivel de log desejado (ex: 'DEBUG', 'INFO', 'WARNING', 'ERROR').
            Caso omitido, utiliza a variavel de ambiente LOG_LEVEL ou 'INFO'.
    """
    resolved_level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, resolved_level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """Retorna uma instancia de logger configurada para o modulo informado.

    Args:
        name: Nome do modulo solicitante (tipicamente __name__).

    Returns:
        logging.Logger: Instancia de logger.
    """
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
