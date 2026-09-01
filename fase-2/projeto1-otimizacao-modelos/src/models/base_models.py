"""Modelos base da Fase 1 e reprodução dos baselines (Etapa 1).

Reutiliza os três modelos da Fase 1 (Regressão Logística, Random Forest e SVC)
e reproduz os resultados originais via ``GridSearchCV`` com os mesmos grids,
salvando-os como **referência** para a comparação com os modelos otimizados
pelo Algoritmo Genético (Etapas 3–4).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data import DATASET_NAME, get_split
from src.evaluate import compute_metrics
from src.preprocessing import build_preprocessor

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_METRICS_DIR = ROOT_DIR / "results" / "metrics"
MODELS_DIR = ROOT_DIR / "models"

for path in (RESULTS_METRICS_DIR, MODELS_DIR):
    path.mkdir(parents=True, exist_ok=True)

# Fábrica de modelos: nome -> função que recebe parâmetros e devolve o estimador.
MODEL_FACTORIES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "logreg": lambda params: LogisticRegression(max_iter=5000, random_state=42, **params),
    "rf": lambda params: RandomForestClassifier(random_state=42, n_jobs=-1, **params),
    "svc": lambda params: SVC(random_state=42, **params),
}

# Mesmos grids da Fase 1 — usados apenas para reproduzir o baseline original.
BASELINE_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logreg": {
        "model__C": [0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    "rf": {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 6, 12],
        "model__min_samples_split": [2, 6],
    },
    "svc": {
        "model__C": [0.5, 1.0, 3.0],
        "model__gamma": ["scale", 0.01, 0.001],
        "model__kernel": ["rbf"],
    },
}


def build_model(model_name: str, params: dict[str, Any] | None = None) -> Any:
    """Constrói um estimador do modelo indicado com os parâmetros fornecidos."""
    return MODEL_FACTORIES[model_name](params or {})


def train_baselines() -> dict[str, Any]:
    """Reproduz os baselines da Fase 1 e salva artefatos de referência.

    Returns:
        Dict com ``best_estimators``, ``results`` (DataFrame) e ``best_name``.
    """
    X_train, X_test, y_train, y_test = get_split()
    preprocessor = build_preprocessor(X_train.columns.tolist())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_estimators: dict[str, Pipeline] = {}
    rows: list[dict[str, Any]] = []

    for name, grid in BASELINE_GRIDS.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", build_model(name)),
        ])
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_estimators[name] = search.best_estimator_

        pred = search.best_estimator_.predict(X_test)
        rows.append({
            "model": name,
            **compute_metrics(y_test, pred),
            "best_cv_accuracy": float(search.best_score_),
            "best_params": str(search.best_params_),
        })

    results_df = (
        pd.DataFrame(rows)
        .sort_values(["accuracy", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    results_df.to_csv(RESULTS_METRICS_DIR / "baseline_model_comparison.csv", index=False)

    best_name = str(results_df.iloc[0]["model"])
    for name, est in best_estimators.items():
        joblib.dump(est, MODELS_DIR / f"baseline_{name}.joblib")
    joblib.dump(best_estimators[best_name], MODELS_DIR / "baseline_best.joblib")

    summary = pd.DataFrame([{
        "dataset": DATASET_NAME,
        "best_model": best_name,
        "accuracy": float(results_df.iloc[0]["accuracy"]),
        "precision": float(results_df.iloc[0]["precision"]),
        "recall": float(results_df.iloc[0]["recall"]),
        "f1": float(results_df.iloc[0]["f1"]),
    }])
    summary.to_csv(RESULTS_METRICS_DIR / "baseline_summary.csv", index=False)

    return {
        "best_estimators": best_estimators,
        "results": results_df,
        "best_name": best_name,
    }
