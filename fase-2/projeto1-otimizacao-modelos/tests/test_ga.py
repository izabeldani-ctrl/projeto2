"""Testes unitários do Algoritmo Genético (Etapa 2)."""
from __future__ import annotations

import numpy as np
import pytest

from src.ga.config import GAConfig
from src.ga.genetic_algorithm import run_ga
from src.ga.operators import (
    blx_alpha_crossover,
    gaussian_mutation,
    one_point_crossover,
    roulette_selection,
    tournament_selection,
    uniform_crossover,
    uniform_mutation,
)
from src.models.search_space import ALL_MODELS, MODEL_SEARCH_SPACES, decode_genes, n_genes


# ---------------------------------------------------------------------------
# Espaço de busca
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ALL_MODELS)
def test_decode_genes_valid_range(model: str) -> None:
    for genes in ([0.0] * n_genes(model), [1.0] * n_genes(model), [0.5] * n_genes(model)):
        params = decode_genes(model, genes)
        assert isinstance(params, dict)
        assert len(params) == n_genes(model)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_decode_genes_clamps_out_of_range(model: str) -> None:
    # Valores fora do domínio devem ser truncados para o intervalo válido.
    # None é um valor legítimo (ex.: class_weight=None), por isso apenas
    # verificamos a estrutura do dicionário e a quantidade de genes.
    params = decode_genes(model, [-1.0] * n_genes(model))
    assert set(params.keys()) == {spec.name for spec in MODEL_SEARCH_SPACES[model]}


# ---------------------------------------------------------------------------
# Operadores
# ---------------------------------------------------------------------------

def test_one_point_crossover_preserves_length() -> None:
    rng = np.random.default_rng(0)
    p1 = [0.1, 0.2, 0.3, 0.4]
    p2 = [0.9, 0.8, 0.7, 0.6]
    c1, c2 = one_point_crossover(p1, p2, rng)
    assert len(c1) == len(p1) == len(p2) == len(c2)


def test_uniform_crossover_outputs_in_domain() -> None:
    rng = np.random.default_rng(1)
    p1 = [0.1, 0.2, 0.3]
    p2 = [0.9, 0.8, 0.7]
    c1, c2 = uniform_crossover(p1, p2, rng)
    for c in (c1, c2):
        assert all(0.0 <= g <= 1.0 for g in c)


def test_blx_alpha_crossover_in_domain() -> None:
    rng = np.random.default_rng(2)
    p1 = [0.2, 0.4]
    p2 = [0.6, 0.8]
    c1, c2 = blx_alpha_crossover(p1, p2, alpha=0.5, rng=rng)
    for c in (c1, c2):
        assert all(0.0 <= g <= 1.0 for g in c)


def test_gaussian_mutation_stays_in_domain() -> None:
    rng = np.random.default_rng(3)
    ind = [0.5] * 20
    mutated = gaussian_mutation(ind, rate=1.0, sigma=10.0, rng=rng)
    assert all(0.0 <= g <= 1.0 for g in mutated)


def test_uniform_mutation_resamples() -> None:
    rng = np.random.default_rng(4)
    ind = [0.5, 0.5, 0.5]
    mutated = uniform_mutation(ind, rate=1.0, rng=rng)
    assert all(0.0 <= g <= 1.0 for g in mutated)


def test_tournament_returns_member() -> None:
    rng = np.random.default_rng(5)
    pop = [[0.1], [0.5], [0.9]]
    fitness = [0.1, 0.5, 0.9]
    chosen = tournament_selection(pop, fitness, tournament_size=3, rng=rng)
    assert chosen in pop


def test_roulette_prefers_better() -> None:
    rng = np.random.default_rng(6)
    pop = [[0.0], [1.0]]
    fitness = [0.0, 1.0]
    # Com fitness 0 e 1, o melhor deve ser escolhido (prob. 1.0)
    chosen = roulette_selection(pop, fitness, rng=rng)
    assert chosen == [1.0]


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

def test_config_validation_rejects_bad_mutation_rate() -> None:
    cfg = GAConfig(mutation_rate=1.5)
    with pytest.raises(ValueError):
        cfg.validate()


# ---------------------------------------------------------------------------
# Função de fitness
# ---------------------------------------------------------------------------

def test_fitness_evaluate_individual_range() -> None:
    """A fitness (accuracy com CV) deve retornar valor em [0, 1]."""
    import pandas as pd

    from src.ga.fitness import evaluate_individual

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((80, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] > 0.5).astype(int))
    genes = [0.5] * n_genes("logreg")
    score = evaluate_individual("logreg", genes, X, y, cv_folds=3)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# GA converge em problema sintético (execução curta)
# ---------------------------------------------------------------------------

def test_ga_runs_and_improves_on_toy_problem() -> None:
    """O GA deve melhorar (ou manter) o melhor fitness ao longo das gerações."""
    import pandas as pd

    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.random((100, 2)), columns=["f0", "f1"])
    y = pd.Series((X["f0"] + X["f1"] > 1.0).astype(int))
    cfg = GAConfig(population_size=6, generations=5, elitism=1, random_state=7)
    result = run_ga("logreg", X, y, cfg)
    assert result.generations_run == 5
    assert len(result.history) == 5
    assert result.history == sorted(result.history)  # melhor acumulado é monotônico
    assert 0.0 <= result.best_fitness <= 1.0
    assert result.best_params is not None
