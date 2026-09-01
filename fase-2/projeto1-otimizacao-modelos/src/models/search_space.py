"""Espaço de busca de hiperparâmetros para o Algoritmo Genético (Etapa 2).

Cada modelo da Fase 1 tem uma lista de genes (hiperparâmetros) que o GA irá
otimizar. A representação interna de cada gene é **normalizada em [0, 1]** para
facilitar os operadores genéticos (crossover/mutação); a função ``decode_genes``
converte o vetor normalizado de volta para os parâmetros reais do scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class GeneSpec:
    """Especificação de um gene (hiperparâmetro).

    Attributes:
        name: nome do parâmetro no scikit-learn (ex.: ``C``, ``max_depth``).
        kind: tipo do gene — ``"float"``, ``"int"`` ou ``"categorical"``.
        bounds: intervalo (min, max) para genes numéricos.
        choices: lista de opções para genes categóricos.
        log_scale: se True, a busca numérica ocorre em escala logarítmica.
    """

    name: str
    kind: str  # "float" | "int" | "categorical"
    bounds: Optional[tuple[float, float]] = None  # para float/int
    choices: Optional[list[Any]] = None  # para categorical
    log_scale: bool = False  # para float/int


MODEL_SEARCH_SPACES: dict[str, tuple[GeneSpec, ...]] = {
    "logreg": (
        GeneSpec(name="C", kind="float", bounds=(1e-4, 1e2), log_scale=True),
        GeneSpec(name="class_weight", kind="categorical", choices=[None, "balanced"]),
    ),
    "rf": (
        # Limite de n_estimators em 200: dataset pequeno (569 amostras) satura
        # bem antes disso e o custo por avaliação de fitness cai drasticamente.
        GeneSpec(name="n_estimators", kind="int", bounds=(20, 200)),
        GeneSpec(name="max_depth", kind="int", bounds=(2, 30)),
        GeneSpec(name="min_samples_split", kind="int", bounds=(2, 20)),
    ),
    "svc": (
        GeneSpec(name="C", kind="float", bounds=(1e-3, 1e3), log_scale=True),
        GeneSpec(name="gamma", kind="float", bounds=(1e-4, 1e1), log_scale=True),
        GeneSpec(name="kernel", kind="categorical", choices=["rbf", "linear", "poly"]),
    ),
}

ALL_MODELS = tuple(MODEL_SEARCH_SPACES.keys())


def n_genes(model_name: str) -> int:
    """Número de genes (hiperparâmetros) do modelo."""
    return len(MODEL_SEARCH_SPACES[model_name])


def decode_gene(spec: GeneSpec, value: float) -> Any:
    """Converte um gene normalizado em [0, 1] para o valor real do parâmetro."""
    value = float(np.clip(value, 0.0, 1.0))

    if spec.kind == "categorical":
        assert spec.choices is not None
        idx = int(round(value * (len(spec.choices) - 1)))
        return spec.choices[idx]

    assert spec.bounds is not None
    low, high = spec.bounds
    if spec.log_scale:
        low, high = np.log10(low), np.log10(high)
        decoded = 10.0 ** (low + value * (high - low))
    else:
        decoded = low + value * (high - low)

    return int(round(decoded)) if spec.kind == "int" else float(decoded)


def decode_genes(model_name: str, genes: list[float]) -> dict[str, Any]:
    """Decodifica um cromossomo (vetor em [0, 1]) em parâmetros do scikit-learn."""
    specs = MODEL_SEARCH_SPACES[model_name]
    if len(genes) != len(specs):
        raise ValueError(
            f"Modelo '{model_name}' espera {len(specs)} genes, recebidos {len(genes)}"
        )
    return {spec.name: decode_gene(spec, value) for spec, value in zip(specs, genes)}


def sample_random_params(model_name: str, rng: Optional[np.random.Generator] = None) -> dict[str, Any]:
    """Amostra parâmetros aleatórios válidos (útil para a população inicial)."""
    rng = rng or np.random.default_rng()
    genes = rng.random(n_genes(model_name)).tolist()
    return decode_genes(model_name, genes)
