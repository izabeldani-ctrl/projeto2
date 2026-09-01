"""Loop principal do Algoritmo Genético.

Fluxo: geração da população inicial → avaliação → elitismo → seleção →
cruzamento → mutação → reavaliação, repetindo por ``generations`` gerações
(ou até a estagnação, se ``patience`` for definido).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.ga.config import GAConfig
from src.ga.fitness import evaluate_population
from src.ga.operators import crossover, mutate, select
from src.models.search_space import decode_genes, n_genes

logger = logging.getLogger("projeto1.ga")


@dataclass
class GAResult:
    """Resultado de uma execução do Algoritmo Genético."""

    model_name: str
    best_genes: list[float]
    best_params: dict[str, object]
    best_fitness: float
    history: list[float]  # melhor fitness acumulado por geração
    mean_history: list[float]  # fitness médio da população por geração
    generations_run: int
    config: GAConfig


def _init_population(
    rng: np.random.Generator,
    size: int,
    n_genes_: int,
) -> list[list[float]]:
    return [rng.random(n_genes_).tolist() for _ in range(size)]


def run_ga(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Optional[GAConfig] = None,
) -> GAResult:
    """Executa o Algoritmo Genético para otimizar os hiperparâmetros do modelo.

    Args:
        model_name: nome do modelo a otimizar.
        X_train: features de treino (usadas na fitness via CV).
        y_train: alvo de treino.
        config: configuração do GA (padrão se não informada).

    Returns:
        :class:`GAResult` com o melhor indivíduo e o histórico de convergência.
    """
    config = config or GAConfig()
    config.validate()
    rng = np.random.default_rng(config.random_state)
    ng = n_genes(model_name)

    logger.info(
        "GA iniciado — modelo=%s, pop=%d, gens=%d, cx=%.2f, mut=%.2f, sel=%s, "
        "elitismo=%d, seed=%d",
        model_name, config.population_size, config.generations,
        config.crossover_rate, config.mutation_rate, config.selection,
        config.elitism, config.random_state,
    )

    # População inicial aleatória
    population = _init_population(rng, config.population_size, ng)
    fitness = evaluate_population(
        model_name, population, X_train, y_train,
        config.cv_folds, config.random_state,
    )

    best_idx = int(np.argmax(fitness))
    best_genes = population[best_idx][:]
    best_fitness = fitness[best_idx]
    stagnation = 0

    history: list[float] = []
    mean_history: list[float] = []

    for gen in range(1, config.generations + 1):
        # Elitismo: preserva os melhores indivíduos sem alteração
        order = np.argsort(fitness)[::-1]
        elites = [population[i][:] for i in order[: config.elitism]]

        # Gera os filhos até completar a nova população
        offspring: list[list[float]] = []
        while len(offspring) < config.population_size:
            p1 = select(population, fitness, config, rng)
            p2 = select(population, fitness, config, rng)
            c1, c2 = crossover(p1, p2, config, rng)
            offspring.append(mutate(c1, config, rng))
            offspring.append(mutate(c2, config, rng))

        population = elites + offspring[: config.population_size - config.elitism]
        fitness = evaluate_population(
            model_name, population, X_train, y_train,
            config.cv_folds, config.random_state,
        )

        cur_idx = int(np.argmax(fitness))
        if fitness[cur_idx] > best_fitness:
            best_fitness = fitness[cur_idx]
            best_genes = population[cur_idx][:]
            stagnation = 0
        else:
            stagnation += 1

        history.append(best_fitness)
        mean_fitness = float(np.mean(fitness))
        mean_history.append(mean_fitness)

        logger.debug(
            "Geração %d/%d — melhor=%.6f, média=%.6f, estagnação=%d",
            gen, config.generations, best_fitness, mean_fitness, stagnation,
        )

        # Parada antecipada por estagnação
        if config.patience is not None and stagnation >= config.patience:
            logger.info("Parada antecipada por estagnação na geração %d.", gen)
            break

    logger.info(
        "GA concluído — melhor fitness=%.6f, params=%s, gerações=%d",
        best_fitness, decode_genes(model_name, best_genes), len(history),
    )

    return GAResult(
        model_name=model_name,
        best_genes=best_genes,
        best_params=decode_genes(model_name, best_genes),
        best_fitness=best_fitness,
        history=history,
        mean_history=mean_history,
        generations_run=len(history),
        config=config,
    )
