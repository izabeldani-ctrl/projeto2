"""Etapa 4 — Comparativo: modelos otimizados (GA) × originais (Fase 1) no holdout.

Avalia no **mesmo conjunto de teste** da Fase 1 (test_size=0.2, random_state=42,
estratificado):
- os modelos baseline originais (GridSearchCV, salvos em ``models/baseline_*.joblib``);
- os melhores indivíduos de cada experimento do GA (parâmetros em
  ``results/metrics/experiments_details.json``).

Saídas:
- ``results/metrics/comparativo_original_vs_ga.csv``
- ``results/figures/comparativo_original_vs_ga.png`` (barras de accuracy)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import get_split
from src.evaluate import compute_metrics
from src.models.base_models import build_model
from src.models.search_space import ALL_MODELS
from src.preprocessing import build_preprocessor

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_METRICS_DIR = ROOT_DIR / "results" / "metrics"
RESULTS_FIGURES_DIR = ROOT_DIR / "results" / "figures"
MODELS_DIR = ROOT_DIR / "models"

for path in (RESULTS_METRICS_DIR, RESULTS_FIGURES_DIR):
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    X_train, X_test, y_train, y_test = get_split()
    details_path = RESULTS_METRICS_DIR / "experiments_details.json"
    if not details_path.exists():
        raise FileNotFoundError(
            "experiments_details.json não encontrado. Rode run_experiments.py antes."
        )
    details = json.loads(details_path.read_text(encoding="utf-8"))

    rows: list[dict] = []
    best_ga: dict[str, tuple[float, Pipeline]] = {}  # model -> (acc, pipeline)

    for model_name in ALL_MODELS:
        # Baseline original (Fase 1, GridSearchCV) no holdout
        base_path = MODELS_DIR / f"baseline_{model_name}.joblib"
        if base_path.exists():
            base_metrics = compute_metrics(
                y_test, joblib.load(base_path).predict(X_test)
            )
        else:
            base_metrics = {}
        rows.append({
            "model": model_name,
            "version": "baseline (GridSearch)",
            "source": "Fase 1",
            **base_metrics,
        })

        # Melhor indivíduo de cada experimento do GA no holdout
        for exp_name, exp_data in details.get(model_name, {}).items():
            best_params = exp_data["best_params"]
            preprocessor = build_preprocessor(X_train.columns.tolist())
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", build_model(model_name, best_params)),
            ])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            metrics = compute_metrics(y_test, pred)
            rows.append({
                "model": model_name,
                "version": f"GA-{exp_name}",
                "source": f"GA ({exp_data['config']['population_size']} pop, "
                          f"{exp_data['config']['generations']} gen, "
                          f"mut {exp_data['config']['mutation_rate']})",
                **metrics,
            })
            # Guarda o melhor GA por modelo (usado nas Etapas 5–6)
            if model_name not in best_ga or metrics["accuracy"] > best_ga[model_name][0]:
                best_ga[model_name] = (metrics["accuracy"], pipe)

    # Salva o melhor modelo otimizado por GA
    for model_name, (_, pipe) in best_ga.items():
        joblib.dump(pipe, MODELS_DIR / f"ga_best_{model_name}.joblib")
    print("Melhores modelos GA salvos em models/ga_best_<modelo>.joblib")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_METRICS_DIR / "comparativo_original_vs_ga.csv", index=False)

    # Gráfico: accuracy por modelo/versão
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    for i, model in enumerate(ALL_MODELS):
        sub = df[df["model"] == model].reset_index(drop=True)
        x = np.arange(len(sub))
        plt.bar(x + i * bar_width, sub["accuracy"], bar_width,
                label=model, alpha=0.85)
        for j, acc in enumerate(sub["accuracy"]):
            plt.text(j + i * bar_width, acc + 0.001, f"{acc:.4f}",
                     ha="center", fontsize=7, rotation=90)
    plt.xticks(np.arange(len(rows) // len(ALL_MODELS)) + bar_width,
               [r["version"] for r in rows[: len(rows) // len(ALL_MODELS)]])
    plt.ylabel("Accuracy (holdout)")
    plt.ylim(0.90, 1.0)
    plt.title("Comparativo: baseline (Fase 1) vs otimizado (GA) — holdout")
    plt.legend(title="Modelo")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES_DIR / "comparativo_original_vs_ga.png", dpi=140)
    plt.close()

    print("=== Comparativo: baseline vs GA (holdout) ===")
    print(df.to_string(index=False))
    print(f"\nSalvo em: {RESULTS_METRICS_DIR / 'comparativo_original_vs_ga.csv'}")


if __name__ == "__main__":
    main()
