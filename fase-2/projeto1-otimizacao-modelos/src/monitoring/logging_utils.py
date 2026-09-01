"""Configuração de logging estruturado para o projeto (Etapa 5).

Fornece :func:`setup_logging`, que cria um logger com formato consistente
(timestamp, nível, módulo e mensagem) e suporte a saída em console e/ou arquivo.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    name: str = "projeto1",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    console: bool = True,
) -> logging.Logger:
    """Configura e retorna um logger com formato estruturado.

    Args:
        name: nome do logger (ex.: ``"projeto1"``).
        level: nível mínimo de log (``logging.INFO``, ``logging.DEBUG``, ...).
        log_file: caminho opcional para também gravar em arquivo.
        console: se True, adiciona handler no console (stdout).

    Returns:
        O logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
