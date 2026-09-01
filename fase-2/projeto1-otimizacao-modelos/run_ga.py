"""Ponto de entrada da Etapa 2 — executa o Algoritmo Genético para um modelo.

Exemplos:
    python run_ga.py --model svc --name exp1
    python run_ga.py --model rf --pop 50 --gens 30 --mut 0.3 --selection roulette
    python run_ga.py --model logreg --crossover blx_alpha --mutation uniform

Saídas:
    - results/metrics/ga_result_<name>.json   (melhor fitness + parâmetros)
    - results/figures/convergence_<name>.png  (curva de convergência)
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Suprime warnings ruidosos do scikit-learn (não afetam os resultados).
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import matplotlib.pyplot as plt

from src.data import get_split
from src.ga.config import GAConfig
from src.ga.genetic_algorithm import run_ga
from src.monitoring.logging_utils import setup_logging
from src.monitoring.tracker import ExperimentTracker

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_METRICS_DIR = ROOT_DIR / "results" / "metrics"
RESULTS_FIGURES_DIR = ROOT_DIR / "results" / "figures"
LOGS_DIR = ROOT_DIR / "logs"

for path in (RESULTS_METRICS_DIR, RESULTS_FIGURES_DIR, LOGS_DIR):
    path.mkdir(parents=True, exist_ok=True)


def plot_convergence(
    history: list[float],
    mean_history: list[float],
    model_name: str,
    out_path: Path,
) -> None:
    """Gera o gráfico de convergência (melhor fitness e média por geração)."""
    plt.figure(figsize=(8, 5))
    plt.plot(history, label="Melhor fitness (acumulado)", marker="o", markersize=3)
    plt.plot(mean_history, label="Fitness médio da população", linestyle="--")
    plt.xlabel("Geração")
    plt.ylabel("Accuracy média (validação cruzada)")
    plt.title(f"Convergência do GA — {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Otimiza hiperparâmetros com Algoritmo Genético.")
    parser.add_argument("--model", required=True, choices=["logreg", "rf", "svc"])
    parser.add_argument("--name", default="experiment", help="Identificador do experimento (usado nos arquivos de saída).")
    parser.add_argument("--pop", type=int, default=20, help="Tamanho da população.")
    parser.add_argument("--gens", type=int, default=30, help="Número de gerações.")
    parser.add_argument("--cx", type=float, default=0.8, help="Taxa de crossover.")
    parser.add_argument("--mut", type=float, default=0.1, help="Taxa de mutação.")
    parser.add_argument("--selection", default="tournament", choices=["tournament", "roulette"])
    parser.add_argument("--crossover", default="uniform", choices=["uniform", "one_point", "blx_alpha"])
    parser.add_argument("--mutation", default="gaussian", choices=["gaussian", "uniform"])
    parser.add_argument("--elitism", type=int, default=2, help="Número de elite preservados.")
    parser.add_argument("--patience", type=int, default=None, help="Parada por estagnação (opcional).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true",
                        help="Habilita log de cada geração (DEBUG).")
    parser.add_argument("--log-file", default=None,
                        help="Caminho do arquivo de log (ex.: logs/ga.log).")
    return parser


def _config_dict(config: GAConfig) -> dict:
    return {
        "population_size": config.population_size,
        "generations": config.generations,
        "crossover_rate": config.crossover_rate,
        "mutation_rate": config.mutation_rate,
        "elitism": config.elitism,
        "selection": config.selection,
        "tournament_size": config.tournament_size,
        "crossover": config.crossover,
        "mutation": config.mutation,
        "patience": config.patience,
        "random_state": config.random_state,
        "cv_folds": config.cv_folds,
    }


def main() -> None:
    args = build_parser().parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.log_file or (LOGS_DIR / "ga.log")
    setup_logging(name="projeto1", level=level, log_file=log_file)

    config = GAConfig(
        population_size=args.pop,
        generations=args.gens,
        crossover_rate=args.cx,
        mutation_rate=args.mut,
        elitism=args.elitism,
        selection=args.selection,
        crossover=args.crossover,
        mutation=args.mutation,
        patience=args.patience,
        random_state=args.seed,
    )

    X_train, X_test, y_train, y_test = get_split()
    print(f"Otimizando '{args.model}' | {config}")
    result = run_ga(args.model, X_train, y_train, config)

    # Tracking da execução (Etapa 5)
    tracker = ExperimentTracker(RESULTS_METRICS_DIR)
    tracker.log_run(
        model=args.model,
        experiment=args.name,
        config=_config_dict(config),
        fitness=result.best_fitness,
        params=result.best_params,
        generations_run=result.generations_run,
    )

    # Persistência dos resultados
    record = {
        "experiment": args.name,
        "model": args.model,
        "best_fitness": result.best_fitness,
        "best_params": result.best_params,
        "generations_run": result.generations_run,
        "history": result.history,
        "mean_history": result.mean_history,
        "config": _config_dict(config),
    }
    json_path = RESULTS_METRICS_DIR / f"ga_result_{args.name}.json"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    fig_path = RESULTS_FIGURES_DIR / f"convergence_{args.name}.png"
    plot_convergence(result.history, result.mean_history, args.model, fig_path)

    print(f"\nMelhor fitness (accuracy CV): {result.best_fitness:.4f}")
    print(f"Melhores parâmetros: {result.best_params}")
    print(f"Gerações executadas: {result.generations_run}")
    print(f"Tracking: {tracker.csv_path}")
    print(f"Log: {log_file}")
    print(f"Resultado salvo em: {json_path}")
    print(f"Gráfico salvo em: {fig_path}")


if __name__ == "__main__":
    main()
