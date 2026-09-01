"""Testes do módulo de monitoramento (Etapa 5/7): logging e tracker."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.monitoring.logging_utils import setup_logging
from src.monitoring.tracker import ExperimentTracker


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def test_setup_logging_creates_file(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "test.log"
    logger = setup_logging(name="teste_log", level=logging.INFO, log_file=log_file)
    logger.info("mensagem de teste")
    # remove o handler para não poluir outros testes
    logger.handlers.clear()
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "mensagem de teste" in content


def test_setup_logging_returns_logger() -> None:
    logger = setup_logging(name="teste_retorno", console=False)
    assert logger is logging.getLogger("teste_retorno")
    assert logger.level == logging.INFO


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

def test_tracker_log_run_writes_files(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path)
    run_id = tracker.log_run(
        model="svc",
        experiment="exp1",
        config={"population_size": 10},
        fitness=0.98,
        params={"C": 1.0},
        generations_run=5,
        metrics={"accuracy": 0.98},
    )
    assert run_id
    assert tracker.csv_path.exists()
    assert tracker.json_path.exists()

    rows = json.loads(tracker.json_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["model"] == "svc"
    assert rows[0]["fitness"] == 0.98
    assert rows[0]["params"]["C"] == 1.0


def test_tracker_acumula_execucoes(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path)
    tracker.log_run(model="logreg", experiment="exp1", config={}, fitness=0.9,
                    params={}, generations_run=3)
    tracker2 = ExperimentTracker(tmp_path)  # novo tracker lê o histórico salvo
    tracker2.log_run(model="rf", experiment="exp2", config={}, fitness=0.95,
                     params={}, generations_run=4)
    rows = json.loads(tracker2.json_path.read_text(encoding="utf-8"))
    assert len(rows) == 2  # histórico acumulado entre instâncias


def test_tracker_csv_resumo(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path)
    tracker.log_run(model="svc", experiment="exp1", config={}, fitness=0.98,
                    params={}, generations_run=5)
    csv_text = tracker.csv_path.read_text(encoding="utf-8")
    assert "model,experiment,fitness" in csv_text
    assert "svc" in csv_text
