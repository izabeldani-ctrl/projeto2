"""Configurações do Algoritmo Genético.

Todos os parâmetros que variam entre experimentos (Etapa 3) são encapsulados
em :class:`GAConfig` — tamanho da população, taxas, operadores e semente —
garantindo reprodutibilidade.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GAConfig:
    """Parâmetros de execução do Algoritmo Genético."""

    population_size: int = 20
    generations: int = 30
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism: int = 2
    selection: str = "tournament"  # "tournament" | "roulette"
    tournament_size: int = 3
    crossover: str = "uniform"  # "uniform" | "one_point" | "blx_alpha"
    mutation: str = "gaussian"  # "gaussian" | "uniform"
    patience: Optional[int] = None  # parada antecipada por estagnação
    random_state: int = 42
    cv_folds: int = 5

    def validate(self) -> None:
        """Valida os campos da configuração, levantando erro se inválidos."""
        if self.population_size <= 0:
            raise ValueError("population_size deve ser > 0")
        if self.generations <= 0:
            raise ValueError("generations deve ser > 0")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate deve estar em [0, 1]")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate deve estar em [0, 1]")
        if not 0 <= self.elitism <= self.population_size:
            raise ValueError("elitism deve estar em [0, population_size]")
        if self.selection not in ("tournament", "roulette"):
            raise ValueError(f"selection inválido: {self.selection}")
        if self.crossover not in ("uniform", "one_point", "blx_alpha"):
            raise ValueError(f"crossover inválido: {self.crossover}")
        if self.mutation not in ("gaussian", "uniform"):
            raise ValueError(f"mutation inválido: {self.mutation}")
        if self.cv_folds < 2:
            raise ValueError("cv_folds deve ser >= 2")
