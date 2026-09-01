"""Carregamento e divisão dos dados (adaptado da Fase 1).

Reutiliza o dataset Breast Cancer Wisconsin (Diagnostic) e padroniza a
divisão treino/teste usada tanto pelos baselines quanto pelo Algoritmo
Genético, garantindo comparabilidade.
"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

DATASET_NAME = "breast_cancer_wisconsin_diagnostic"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Carrega o dataset completo (X, y)."""
    bunch = load_breast_cancer(as_frame=True)
    return bunch.data.copy(), bunch.target.copy()


def get_split(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divisão estratificada padrão do projeto: X_train, X_test, y_train, y_test."""
    X, y = load_data()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
