"""Função de aptidão (fitness) do Algoritmo Genético.

A aptidão de um cromossomo é a **accuracy média** obtida com validação cruzada
estratificada (5 dobras por padrão), a mesma estratégia usada na Fase 1 — o que
garante comparabilidade direta com os baselines (Etapa 4).
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.models.base_models import build_model
from src.models.search_space import decode_genes
from src.preprocessing import build_preprocessor


def evaluate_individual(
    model_name: str,
    genes: list[float],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> float:
    """Retorna a accuracy média (CV estratificada) de um cromossomo.

    Args:
        model_name: nome do modelo (``"logreg"``, ``"rf"`` ou ``"svc"``).
        genes: cromossomo normalizado em [0, 1].
        X_train: features de treino.
        y_train: alvo de treino.
        cv_folds: número de dobras da validação cruzada.
        random_state: semente para a divisão estratificada (reprodutibilidade).

    Returns:
        Accuracy média entre as dobras.
    """
    params = decode_genes(model_name, genes)
    preprocessor = build_preprocessor(X_train.columns.tolist())
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", build_model(model_name, params)),
    ])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
    return float(scores.mean())


def evaluate_population(
    model_name: str,
    population: list[list[float]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> list[float]:
    """Avalia todos os indivíduos da população (lista de fitness)."""
    return [
        evaluate_individual(model_name, ind, X_train, y_train, cv_folds, random_state)
        for ind in population
    ]
