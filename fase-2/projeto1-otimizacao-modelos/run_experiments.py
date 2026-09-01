"""Etapa 3 — Executa a matriz de experimentos do GA para os modelos.

Roda, para cada modelo, uma série de experimentos com configurações distintas
(tamanho da população, taxas, operadores de seleção/mutação) e salva:

- ``results/metrics/experiments_summary.csv``  — resumo por modelo/experimento
- ``results/metrics/experiments_details.json`` — históricos de convergência
- ``results/figures/convergence_<modelo>_<exp>.png`` — curva individual
- ``results/figures/experiments_<modelo>.png`` — todas as curvas em um gráfico

Exemplos:
    python run_experiments.py                          # todos os modelos
    python run_experiments.py --models logreg svc      # apenas alguns
    python run_experiments.py --exps exp1 exp2         # apenas alguns experimentos
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Suprime warnings ruidosos do scikit-learn (ex.: paralelismo aninhado),
# que não afetam os resultados mas degradam o desempenho e o log.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import pandas as pd

from src.data import get_split
from src.ga.config import GAConfig
from src.ga.genetic_algorithm import run_ga
from src.ga.plotting import plot_convergence, plot_multi_convergence
from src.monitoring.logging_utils import setup_logging

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_METRICS_DIR = ROOT_DIR / "results" / "metrics"
RESULTS_FIGURES_DIR = ROOT_DIR / "results" / "figures"
LOGS_DIR = ROOT_DIR / "logs"

for path in (RESULTS_METRICS_DIR, RESULTS_FIGURES_DIR):
    path.mkdir(parents=True, exist_ok=True)

# Matriz de experimentos — varia população, gerações, mutação e seleção.
# Todas usam a mesma semente para comparação controlada (mesmo ponto de partida).
EXPERIMENTS: dict[str, GAConfig] = {
    "exp1": GAConfig(population_size=10, generations=15, mutation_rate=0.1,
                     selection="tournament", tournament_size=3),
    "exp2": GAConfig(population_size=16, generations=15, mutation_rate=0.1,
                     selection="tournament", tournament_size=3),
    "exp3": GAConfig(population_size=10, generations=20, mutation_rate=0.3,
                     selection="roulette", tournament_size=3),
    "exp4": GAConfig(population_size=16, generations=20, mutation_rate=0.15,
                     selection="tournament", tournament_size=5),
}


def _load_baseline_cv() -> dict[str, float]:
    """Carrega a accuracy média de CV dos baselines (Fase 1) para referência."""
    path = RESULTS_METRICS_DIR / "baseline_model_comparison.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["model"], df["best_cv_accuracy"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa a matriz de experimentos do GA.")
    parser.add_argument("--models", nargs="+", default=["logreg", "rf", "svc"],
                        choices=["logreg", "rf", "svc"])
    parser.add_argument("--exps", nargs="+", default=list(EXPERIMENTS),
                        choices=list(EXPERIMENTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Fator 0-1 que reduz população/gerações (útil p/ modelos lentos).")
    return parser


def _scaled_config(exp_name: str, scale: float) -> GAConfig:
    """Aplica o fator de escala à configuração do experimento."""
    base = EXPERIMENTS[exp_name]
    return GAConfig(
        population_size=max(4, int(round(base.population_size * scale))),
        generations=max(5, int(round(base.generations * scale))),
        crossover_rate=base.crossover_rate,
        mutation_rate=base.mutation_rate,
        elitism=max(1, int(round(base.elitism * scale))),
        selection=base.selection,
        tournament_size=base.tournament_size,
        crossover=base.crossover,
        mutation=base.mutation,
        random_state=base.random_state,
        cv_folds=base.cv_folds,
    )


def main() -> None:
    args = build_parser().parse_args()
    # Logging estruturado das execuções do GA (Etapa 5) — INFO em console, arquivo completo.
    setup_logging(name="projeto1", level=logging.INFO, log_file=LOGS_DIR / "experiments.log")

    baseline_cv = _load_baseline_cv()

    X_train, X_test, y_train, y_test = get_split()
    all_rows: list[dict] = []
    details: dict[str, dict] = {}

    for model in args.models:
        model_rows: list[dict] = []
        model_details: dict = {}
        for exp_name in args.exps:
            cfg = _scaled_config(exp_name, args.scale)
            cfg.random_state = args.seed
            print(f"[{model}] {exp_name}: pop={cfg.population_size}, "
                  f"gens={cfg.generations}, mut={cfg.mutation_rate}, "
                  f"sel={cfg.selection}")
            result = run_ga(model, X_train, y_train, cfg)

            plot_convergence(
                result.history,
                result.mean_history,
                model,
                RESULTS_FIGURES_DIR / f"convergence_{model}_{exp_name}.png",
                title_suffix=f" ({exp_name})",
            )
            model_details[exp_name] = {
                "best_fitness": result.best_fitness,
                "best_params": result.best_params,
                "history": result.history,
                "mean_history": result.mean_history,
                "generations_run": result.generations_run,
                "config": {
                    "population_size": cfg.population_size,
                    "generations": cfg.generations,
                    "crossover_rate": cfg.crossover_rate,
                    "mutation_rate": cfg.mutation_rate,
                    "elitism": cfg.elitism,
                    "selection": cfg.selection,
                    "tournament_size": cfg.tournament_size,
                    "crossover": cfg.crossover,
                    "mutation": cfg.mutation,
                    "random_state": cfg.random_state,
                    "cv_folds": cfg.cv_folds,
                },
            }
            model_rows.append({
                "model": model,
                "experiment": exp_name,
                "population_size": cfg.population_size,
                "generations": cfg.generations,
                "mutation_rate": cfg.mutation_rate,
                "selection": cfg.selection,
                "tournament_size": cfg.tournament_size,
                "best_fitness": round(result.best_fitness, 6),
                "best_params": str(result.best_params),
                "generations_run": result.generations_run,
                "baseline_cv_accuracy": round(float(baseline_cv.get(model, float("nan"))), 6),
            })

        plot_multi_convergence(
            {e: model_details[e]["history"] for e in model_details},
            model,
            RESULTS_FIGURES_DIR / f"experiments_{model}.png",
        )
        all_rows.extend(model_rows)
        details[model] = model_details

    new_df = pd.DataFrame(all_rows)

    # Mescla com resultados existentes, permitindo rodar modelos separadamente
    # sem sobrescrever os anteriores (dedupe por modelo+experimento, mantém o último).
    summary_path = RESULTS_METRICS_DIR / "experiments_summary.csv"
    details_path = RESULTS_METRICS_DIR / "experiments_details.json"
    if summary_path.exists():
        combined = pd.concat([pd.read_csv(summary_path), new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["model", "experiment"], keep="last")
    else:
        combined = new_df
    combined.to_csv(summary_path, index=False)

    old_details: dict = {}
    if details_path.exists():
        old_details = json.loads(details_path.read_text(encoding="utf-8"))
    old_details.update(details)
    details_path.write_text(
        json.dumps(old_details, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Resumo dos experimentos ===")
    df = combined
    show = df[["model", "experiment", "population_size", "generations",
               "mutation_rate", "selection", "best_fitness",
               "baseline_cv_accuracy"]]
    print(show.to_string(index=False))
    print(f"\nSalvo em: {summary_path}")


if __name__ == "__main__":
    main()
