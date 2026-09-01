"""Pré-processamento de dados (adaptado da Fase 1).

Pipeline numérica: imputação pela mediana + padronização (StandardScaler),
aplicada a todas as features (dataset 100% numérico).
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, feature_names),
    ])
