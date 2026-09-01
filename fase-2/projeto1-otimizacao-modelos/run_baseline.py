"""Ponto de entrada da Etapa 1 — reproduz os baselines da Fase 1.

Uso:
    python run_baseline.py

Saídas:
    - results/metrics/baseline_model_comparison.csv
    - results/metrics/baseline_summary.csv
    - models/baseline_<modelo>.joblib e models/baseline_best.joblib
"""
from src.models.base_models import train_baselines


if __name__ == "__main__":
    result = train_baselines()

    print("\nComparação entre baselines (referência para o GA):")
    cols = ["model", "accuracy", "precision", "recall", "f1", "best_cv_accuracy"]
    print(result["results"][cols].to_string(index=False))
    print(f"\nMelhor baseline: {result['best_name']}")
    print("Artefatos salvos em results/metrics/ e models/.")
