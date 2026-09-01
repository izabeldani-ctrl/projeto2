"""Tracking de execuções do GA — persistência em CSV + JSON (Etapa 5).

O :class:`ExperimentTracker` registra cada execução do Algoritmo Genético
(experimento, configuração, fitness, parâmetros e métricas) e mantém um
histórico acumulado em ``results/metrics/ga_runs.csv`` e
``results/metrics/ga_runs.json``.
"""
from __future__ import annotations

import csv
import json
import time
import uuid
from pathlib import Path


class ExperimentTracker:
    """Registra execuções do GA e persiste em CSV e JSON (append)."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "ga_runs.csv"
        self.json_path = self.output_dir / "ga_runs.json"
        self._runs: list[dict] = []
        if self.json_path.exists():
            try:
                self._runs = json.loads(self.json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._runs = []

    def log_run(
        self,
        *,
        model: str,
        experiment: str,
        config: dict,
        fitness: float,
        params: dict,
        generations_run: int,
        metrics: dict | None = None,
        run_id: str | None = None,
    ) -> str:
        """Registra uma execução e atualiza os arquivos CSV/JSON.

        Returns:
            O ``run_id`` gerado (ou informado).
        """
        run_id = run_id or uuid.uuid4().hex[:8]
        record = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "experiment": experiment,
            "fitness": float(fitness),
            "generations_run": int(generations_run),
            "params": params,
            "config": config,
            "metrics": metrics or {},
        }
        self._runs.append(record)
        self._write_csv()
        self._write_json()
        return run_id

    def _write_csv(self) -> None:
        if not self._runs:
            return
        fields = ["run_id", "timestamp", "model", "experiment", "fitness", "generations_run"]
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in self._runs:
                writer.writerow({k: r.get(k) for k in fields})

    def _write_json(self) -> None:
        self.json_path.write_text(
            json.dumps(self._runs, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @property
    def runs(self) -> list[dict]:
        return self._runs
