"""Etapa 6 — Demonstração da integração com LLM (Ollama + llama3.2).

Fluxo:
1. Carrega o melhor modelo otimizado por GA (LogReg) e o baseline da Fase 1.
2. Prevê no conjunto de teste e seleciona casos representativos (benigno,
   maligno e caso-limite).
3. Gera explicações em linguagem natural via Ollama (prompt engineering).
4. Gera uma interpretação agregada do modelo (métricas + ganhos vs baseline).
5. Avalia a qualidade das respostas (checklist) e salva tudo em markdown/JSON.

Uso:
    python run_llm_demo.py            # usa o modelo padrão (llama3.2)
    python run_llm_demo.py --model llama3.2
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import get_split
from src.evaluate import compute_metrics
from src.llm.interpreter import (
    DEFAULT_MODEL,
    OllamaClient,
    evaluate_explanation,
    interpret_case,
    interpret_model,
    translate_feature_name,
)
from src.monitoring.logging_utils import setup_logging

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
RESULTS_METRICS_DIR = ROOT_DIR / "results" / "metrics"
LOGS_DIR = ROOT_DIR / "logs"

for path in (REPORTS_DIR, LOGS_DIR):
    path.mkdir(parents=True, exist_ok=True)

MODEL_LABEL = "Regressão Logística otimizada por GA"
N_TOP_FEATURES = 5


def _top_features_for_case(
    pipeline, row: pd.Series, n: int = N_TOP_FEATURES
) -> list[tuple[str, float]]:
    """Features mais relevantes de um caso (LogReg): contribuição |coef| × valor."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    scaled = preprocessor.transform(row.to_frame().T)
    contrib = model.coef_[0] * scaled[0]
    idx = np.argsort(np.abs(contrib))[::-1][:n]
    return [(row.index[i], float(row.iloc[i])) for i in idx]


def _select_cases(
    X_test: pd.DataFrame, y_test: pd.Series, pred: np.ndarray, proba: np.ndarray
) -> list[tuple[int, str]]:
    """Seleciona 3 casos representativos: benigno, maligno e caso-limite."""
    y = y_test.to_numpy()
    benign_correct = np.where((pred == 0) & (y == 0))[0]
    malign_correct = np.where((pred == 1) & (y == 1))[0]

    cases: list[tuple[int, str]] = []
    if len(benign_correct):
        pos = benign_correct[int(np.argmin(proba[benign_correct]))]
        cases.append((int(pos), "Caso benigno (maior confiança)"))
    if len(malign_correct):
        pos = malign_correct[int(np.argmax(proba[malign_correct]))]
        cases.append((int(pos), "Caso maligno (maior confiança)"))
    # Caso-limite: decisão mais próxima de 50%
    limiar = int(np.argmin(np.abs(proba - 0.5)))
    if limiar not in [c[0] for c in cases]:
        cases.append((limiar, "Caso-limite (maior incerteza)"))
    return cases[:3]


def _re_evaluate() -> None:
    """Re-avalia as explicações salvas (sem chamar a LLM) — útil após ajustar o avaliador."""
    out_json = REPORTS_DIR / "llm_output.json"
    out_md = REPORTS_DIR / "llm_output.md"
    if not out_json.exists():
        print("llm_output.json não encontrado. Rode a demonstração completa primeiro.")
        return
    outputs = json.loads(out_json.read_text(encoding="utf-8"))

    md_lines = ["# Demonstração — Integração com LLM (Etapa 6)", "", "---", ""]
    for o in outputs:
        if o["caso"] == "Interpretação agregada do modelo":
            aval = evaluate_explanation(o["explicacao"], expected_features=[])
        else:
            feats = [f["feature"] for f in o["features_topo"]]
            feats += [translate_feature_name(f) for f in feats[: len(o["features_topo"])]]
            aval = evaluate_explanation(
                o["explicacao"],
                expected_classe=o["previsao"],
                expected_features=feats,
            )
        o["avaliacao"] = aval
        md_lines += [
            f"## {o['caso']}",
            "",
            "**Explicação gerada pela LLM:**",
            "",
            o["explicacao"],
            "",
            f"**Avaliação (checklist):** score {aval['score_total']}/1.0 — "
            + "; ".join(f"{k}={v}" for k, v in aval.items()
                        if k not in ("score_total", "num_palavras")),
            "",
            "---",
            "",
        ]
    out_json.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Re-avaliação concluída. Atualizados: {out_json} e {out_md}")
    for o in outputs:
        print(f"- [{o['caso']}] score qualidade: {o['avaliacao']['score_total']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstra a integração LLM (Etapa 6).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Nome do modelo no Ollama.")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--re-evaluate", action="store_true",
                        help="Re-avalia as explicações salvas sem chamar a LLM.")
    args = parser.parse_args()

    if args.re_evaluate:
        _re_evaluate()
        return

    setup_logging(name="projeto1", level=logging.INFO, log_file=LOGS_DIR / "llm_demo.log")
    client = OllamaClient(model=args.model)

    if not client.is_available():
        print("\n[AVISO] Ollama não está acessível em localhost:11434.")
        print("1) Confirme que o Ollama está instalado e rodando.")
        print("2) Baixe o modelo:  ollama pull llama3.2")
        return

    X_train, X_test, y_train, y_test = get_split()
    pipe = joblib.load(MODELS_DIR / "ga_best_logreg.joblib")
    base = joblib.load(MODELS_DIR / "baseline_logreg.joblib")

    Xt = X_test.reset_index(drop=True)
    pred = pipe.predict(Xt)
    proba = pipe.predict_proba(Xt)[:, 1]
    metrics = compute_metrics(y_test.reset_index(drop=True), pred)
    base_metrics = compute_metrics(y_test.reset_index(drop=True), base.predict(Xt))
    gains = {k: metrics[k] - base_metrics[k] for k in ("accuracy", "precision", "recall", "f1")}

    model = pipe.named_steps["model"]
    best_params = {k: model.get_params()[k] for k in ("C", "class_weight") if k in model.get_params()}

    print(f"Modelo: {MODEL_LABEL} | métricas { {k: round(v, 4) for k, v in metrics.items()} }")
    print(f"Gerando explicações com Ollama ({args.model})...")

    outputs: list[dict] = []
    md_lines: list[str] = [
        "# Demonstração — Integração com LLM (Etapa 6)",
        "",
        f"- Modelo de ML: **{MODEL_LABEL}**",
        f"- LLM: **{args.model}** (Ollama local)",
        "",
        "---",
        "",
    ]

    # ---- Explicações individuais ----
    for pos, descricao in _select_cases(Xt, y_test.reset_index(drop=True), pred, proba):
        row = Xt.iloc[pos]
        top = _top_features_for_case(pipe, row)
        texto = interpret_case(
            client,
            model_name=MODEL_LABEL,
            prediction=int(pred[pos]),
            proba_maligna=float(proba[pos]),
            top_features=top,
        )
        aval = evaluate_explanation(
            texto,
            expected_classe="maligno" if pred[pos] == 1 else "benigno",
            expected_features=[f[0] for f in top]
            + [translate_feature_name(f[0]) for f in top],  # inglês + português
        )
        outputs.append({
            "caso": descricao,
            "posicao": pos,
            "classe_real": "maligno" if y_test.iloc[pos] == 1 else "benigno",
            "previsao": "maligno" if pred[pos] == 1 else "benigno",
            "probabilidade_maligna": round(float(proba[pos]), 4),
            "features_topo": [{"feature": n, "valor": round(v, 3)} for n, v in top],
            "explicacao": texto,
            "avaliacao": aval,
        })
        md_lines += [
            f"## {descricao}",
            f"- Real: **{outputs[-1]['classe_real']}** | Previsto: **{outputs[-1]['previsao']}** | "
            f"Prob. malignidade: **{proba[pos]*100:.1f}%**",
            f"- Features mais relevantes: " + "; ".join(f"{n} = {v:.2f}" for n, v in top),
            "",
            "**Explicação gerada pela LLM:**",
            "",
            texto,
            "",
            f"**Avaliação (checklist):** score {aval['score_total']}/1.0 — " + "; ".join(
                f"{k}={v}" for k, v in aval.items() if k not in ("score_total", "num_palavras")
            ),
            "",
            "---",
            "",
        ]

    # ---- Interpretação agregada ----
    modelo_texto = interpret_model(
        client,
        model_name=MODEL_LABEL,
        metrics=metrics,
        best_params=best_params,
        gains=gains,
    )
    aval_modelo = evaluate_explanation(modelo_texto, expected_features=[])
    outputs.append({
        "caso": "Interpretação agregada do modelo",
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "gains_vs_baseline": {k: round(v, 4) for k, v in gains.items()},
        "best_params": best_params,
        "explicacao": modelo_texto,
        "avaliacao": aval_modelo,
    })
    md_lines += [
        "## Interpretação agregada do modelo",
        f"- Métricas no teste: { {k: round(v, 4) for k, v in metrics.items()} }",
        f"- Ganhos vs baseline: { {k: round(v, 4) for k, v in gains.items()} }",
        "",
        "**Explicação gerada pela LLM:**",
        "",
        modelo_texto,
        "",
        f"**Avaliação:** score {aval_modelo['score_total']}/1.0",
        "",
    ]

    # ---- Persistência ----
    out_md = REPORTS_DIR / "llm_output.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    out_json = REPORTS_DIR / "llm_output.json"
    out_json.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSalvo em: {out_md}")
    print(f"Salvo em: {out_json}")
    for o in outputs:
        print(f"- [{o['caso']}] score qualidade: {o['avaliacao']['score_total']}")


if __name__ == "__main__":
    main()
