"""Operadores genéticos: seleção, cruzamento e mutação.

Todos os indivíduos são cromossomos de valores reais em [0, 1]
(ver ``src.models.search_space``). Os operadores garantem que os filhos
permaneçam no domínio válido (valores fora de [0, 1] são truncados).
"""
from __future__ import annotations

import numpy as np

from src.ga.config import GAConfig


# ---------------------------------------------------------------------------
# Seleção
# ---------------------------------------------------------------------------

def tournament_selection(
    population: list[list[float]],
    fitness: list[float],
    tournament_size: int,
    rng: np.random.Generator,
) -> list[float]:
    """Seleção por torneio: sorteia ``k`` indivíduos e retorna o de maior fitness."""
    idx = rng.integers(0, len(population), size=tournament_size)
    best = max(idx, key=lambda i: fitness[i])
    return population[best]


def roulette_selection(
    population: list[list[float]],
    fitness: list[float],
    rng: np.random.Generator,
) -> list[float]:
    """Seleção por roleta: probabilidade proporcional ao fitness (relativo)."""
    f = np.asarray(fitness, dtype=float)
    f = f - f.min()  # desloca para >= 0 (roleta)
    total = f.sum()
    if total <= 0:
        idx = int(rng.integers(0, len(population)))
        return population[idx]
    idx = int(rng.choice(len(population), p=f / total))
    return population[idx]


def select(
    population: list[list[float]],
    fitness: list[float],
    config: GAConfig,
    rng: np.random.Generator,
) -> list[float]:
    """Seleciona um progenitor conforme a estratégia da configuração."""
    if config.selection == "tournament":
        return tournament_selection(population, fitness, config.tournament_size, rng)
    return roulette_selection(population, fitness, rng)


# ---------------------------------------------------------------------------
# Cruzamento
# ---------------------------------------------------------------------------

def one_point_crossover(
    p1: list[float],
    p2: list[float],
    rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    """Crossover de 1 ponto: troca a cauda dos cromossomos a partir de um ponto."""
    point = int(rng.integers(1, len(p1)))
    return p1[:point] + p2[point:], p2[:point] + p1[point:]


def uniform_crossover(
    p1: list[float],
    p2: list[float],
    rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    """Crossover uniforme: cada gene é sorteado entre os dois pais."""
    mask = rng.random(len(p1)) < 0.5
    c1 = [p1[i] if mask[i] else p2[i] for i in range(len(p1))]
    c2 = [p2[i] if mask[i] else p1[i] for i in range(len(p1))]
    return c1, c2


def blx_alpha_crossover(
    p1: list[float],
    p2: list[float],
    alpha: float,
    rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    """Crossover BLX-α (contínuo): filho amostrado no intervalo expandido dos pais."""
    c1: list[float] = []
    c2: list[float] = []
    for a, b in zip(p1, p2):
        lo, hi = min(a, b), max(a, b)
        span = hi - lo
        lo -= alpha * span
        hi += alpha * span
        c1.append(float(np.clip(rng.uniform(lo, hi), 0.0, 1.0)))
        c2.append(float(np.clip(rng.uniform(lo, hi), 0.0, 1.0)))
    return c1, c2


def crossover(
    p1: list[float],
    p2: list[float],
    config: GAConfig,
    rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    """Aplica o cruzamento (ou copia os pais) conforme a taxa e a estratégia."""
    if rng.random() > config.crossover_rate:
        return p1[:], p2[:]
    if config.crossover == "one_point":
        return one_point_crossover(p1, p2, rng)
    if config.crossover == "blx_alpha":
        return blx_alpha_crossover(p1, p2, alpha=0.5, rng=rng)
    return uniform_crossover(p1, p2, rng)


# ---------------------------------------------------------------------------
# Mutação
# ---------------------------------------------------------------------------

def gaussian_mutation(
    individual: list[float],
    rate: float,
    sigma: float = 0.2,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Mutação gaussiana: perturba cada gene com ruído N(0, sigma) e trunca em [0, 1]."""
    rng = rng or np.random.default_rng()
    return [
        float(np.clip(g + rng.normal(0.0, sigma), 0.0, 1.0)) if rng.random() < rate else g
        for g in individual
    ]


def uniform_mutation(
    individual: list[float],
    rate: float,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Mutação uniforme: ressorta o gene para um valor aleatório em [0, 1]."""
    rng = rng or np.random.default_rng()
    return [float(rng.random()) if rng.random() < rate else g for g in individual]


def mutate(
    individual: list[float],
    config: GAConfig,
    rng: np.random.Generator,
) -> list[float]:
    """Aplica a mutação conforme a estratégia da configuração."""
    if config.mutation == "uniform":
        return uniform_mutation(individual, config.mutation_rate, rng=rng)
    return gaussian_mutation(individual, config.mutation_rate, rng=rng)
