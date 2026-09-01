"""Gráficos de convergência do Algoritmo Genético (Etapa 3)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_convergence(
    history: list[float],
    mean_history: list[float],
    model_name: str,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Gera o gráfico de convergência (melhor fitness e média por geração)."""
    plt.figure(figsize=(8, 5))
    plt.plot(history, label="Melhor fitness (acumulado)", marker="o", markersize=3)
    plt.plot(mean_history, label="Fitness médio da população", linestyle="--")
    plt.xlabel("Geração")
    plt.ylabel("Accuracy média (validação cruzada)")
    plt.title(f"Convergência do GA — {model_name}{title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_multi_convergence(
    histories: dict[str, list[float]],
    model_name: str,
    out_path: Path,
) -> None:
    """Compara as curvas de convergência de vários experimentos em um só gráfico."""
    plt.figure(figsize=(8, 5))
    for exp, hist in histories.items():
        plt.plot(hist, label=exp, marker="o", markersize=3)
    plt.xlabel("Geração")
    plt.ylabel("Melhor fitness (accuracy média CV)")
    plt.title(f"Convergência por experimento — {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
