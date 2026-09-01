"""Integração com LLM local via Ollama — interpretação de diagnósticos (Etapa 6).

Módulo com:
- :class:`OllamaClient`: cliente HTTP para a API do Ollama (sem dependências).
- :func:`build_individual_prompt`: prompt para explicar um caso (diagnóstico).
- :func:`build_model_summary_prompt`: prompt para interpretar o modelo como um todo.
- :func:`interpret_case` / :func:`interpret_model`: fluxos prontos de geração.
- :func:`evaluate_explanation`: checklist de qualidade da resposta gerada.

Prompt engineering: prompt de sistema com papel clínico e restrições (não
inventar dados, não substituir o médico, resposta em português) + contexto
estruturado com os números reais do modelo.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional
from urllib import error, request

logger = logging.getLogger("projeto1.llm")

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_TIMEOUT = 600  # 1ª chamada carrega o modelo na RAM (lento em CPU)

# Tradução de tokens dos nomes de features do Breast Cancer para português
_FEATURE_TRANSLATIONS = {
    "mean": "média", "error": "erro", "worst": "pior",
    "radius": "raio", "texture": "textura", "perimeter": "perímetro",
    "area": "área", "smoothness": "suavidade", "compactness": "compacidade",
    "concavity": "concavidade", "concave": "côncavos", "points": "pontos",
    "symmetry": "simetria", "fractal": "fractal", "dimension": "dimensão",
}


def translate_feature_name(name: str) -> str:
    """Traduz um nome de feature do sklearn para português (ex.: 'worst area' → 'pior área')."""
    tokens = name.replace("_", " ").split()
    translated: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "concave" and i + 1 < len(tokens) and tokens[i + 1] == "points":
            translated.append("pontos côncavos")
            i += 2
            continue
        if token == "fractal" and i + 1 < len(tokens) and tokens[i + 1] == "dimension":
            translated.append("dimensão fractal")
            i += 2
            continue
        translated.append(_FEATURE_TRANSLATIONS.get(token, token))
        i += 1
    return " ".join(translated)

SYSTEM_PROMPT = (
    "Você é um assistente de IA de suporte ao diagnóstico em um hospital "
    "universitário. Você recebe saídas de um modelo de machine learning que "
    "classifica tumores de mama como benignos (0) ou malignos (1).\n"
    "SUAS TAREFAS:\n"
    "1. Explicar o resultado do modelo em linguagem natural, clara e objetiva, "
    "para profissionais de saúde.\n"
    "2. Transformar os números (probabilidade, features) em insights acionáveis.\n"
    "REGRAS OBRIGATÓRIAS:\n"
    "- NUNCA invente ou altere dados; use apenas os números fornecidos no contexto.\n"
    "- Deixe sempre claro que o modelo é uma ferramenta de APOIO e não substitui "
    "a avaliação clínica do médico.\n"
    "- Não dê diagnóstico definitivo nem recomendação terapêutica.\n"
    "- Responda apenas em português.\n"
    "- Seja conciso: no máximo 4 a 6 frases."
)


class OllamaClient:
    """Cliente simples para a API HTTP do Ollama (std-lib, sem dependências)."""

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        """Verifica se o servidor Ollama está no ar."""
        try:
            with request.urlopen(f"{self.host}/api/tags", timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 600,
        retries: int = 2,
        fallback: Optional[str] = None,
    ) -> str:
        """Gera texto via ``/api/generate`` (stream desativado), com retry.

        Args:
            prompt: mensagem do usuário.
            system: prompt de sistema opcional.
            temperature: temperatura (0 = determinístico, 1 = criativo).
            max_tokens: limite de tokens de saída.
            retries: tentativas em caso de falha.
            fallback: mensagem retornada em caso de falha persistente.

        Returns:
            Texto gerado pelo modelo (ou o fallback).
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system

        for attempt in range(retries + 1):
            try:
                req = request.Request(
                    f"{self.host}/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text = (data.get("response") or "").strip()
                if text:
                    return text
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha na chamada ao Ollama (tentativa %d): %s", attempt + 1, exc)
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
        return fallback if fallback is not None else "[ERRO] Não foi possível gerar a explicação."


def _fmt_features(top_features: list[tuple[str, float]]) -> str:
    """Formata as features com nome traduzido para português."""
    return "; ".join(
        f"{translate_feature_name(name)} = {value:.3f}" for name, value in top_features
    )


def build_individual_prompt(
    *,
    model_name: str,
    prediction: int,
    proba_maligna: float,
    top_features: list[tuple[str, float]],
) -> str:
    """Monta o prompt de interpretação de um caso individual.

    Args:
        model_name: nome do modelo (ex.: "LogReg otimizada por GA").
        prediction: classe prevista (0 = benigno, 1 = maligno).
        proba_maligna: probabilidade prevista da classe maligna (0–1).
        top_features: lista de (nome, valor) das features mais relevantes do caso.
    """
    classe = "MALIGNO" if prediction == 1 else "BENIGNO"
    features_txt = _fmt_features(top_features) if top_features else "não informadas"
    return (
        f"Contexto do modelo:\n"
        f"- Modelo: {model_name}\n"
        f"- Classificação prevista: {classe}\n"
        f"- Probabilidade de malignidade: {proba_maligna*100:.1f}%\n"
        f"- Principais características do exame: {features_txt}\n\n"
        f"Com base nesses dados, explique o resultado em linguagem natural para "
        f"um profissional de saúde, destacando o nível de risco, os principais "
        f"fatores que influenciaram a decisão e o que isso significa na prática."
    )


def build_model_summary_prompt(
    *,
    model_name: str,
    metrics: dict[str, float],
    best_params: dict[str, Any],
    gains: Optional[dict[str, float]] = None,
) -> str:
    """Monta o prompt de interpretação agregada (do modelo como um todo)."""
    metrics_txt = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    params_txt = json.dumps(best_params, ensure_ascii=False)
    gains_txt = ""
    if gains:
        gains_txt = " Ganho vs baseline: " + ", ".join(
            f"{k} {v:+.4f}" for k, v in gains.items()
        ) + "."
    return (
        f"Modelo: {model_name}\n"
        f"Métricas no conjunto de teste: {metrics_txt}.{gains_txt}\n"
        f"Hiperparâmetros otimizados: {params_txt}\n\n"
        f"Interprete o desempenho desse modelo para um profissional de saúde: o que "
        f"as métricas indicam sobre confiabilidade, pontos fortes e limitações, e "
        f"como o modelo deve ser usado no apoio à triagem."
    )


def interpret_case(
    client: OllamaClient,
    *,
    model_name: str,
    prediction: int,
    proba_maligna: float,
    top_features: list[tuple[str, float]],
) -> str:
    """Gera a explicação em linguagem natural de um caso."""
    prompt = build_individual_prompt(
        model_name=model_name,
        prediction=prediction,
        proba_maligna=proba_maligna,
        top_features=top_features,
    )
    return client.generate(prompt, system=SYSTEM_PROMPT)


def interpret_model(
    client: OllamaClient,
    *,
    model_name: str,
    metrics: dict[str, float],
    best_params: dict[str, Any],
    gains: Optional[dict[str, float]] = None,
) -> str:
    """Gera a interpretação agregada do desempenho do modelo."""
    prompt = build_model_summary_prompt(
        model_name=model_name, metrics=metrics, best_params=best_params, gains=gains
    )
    return client.generate(prompt, system=SYSTEM_PROMPT)


def evaluate_explanation(
    text: str,
    *,
    expected_classe: Optional[str] = None,
    expected_proba: Optional[float] = None,
    expected_features: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Checklist de qualidade da explicação gerada (Etapa 6).

    Critérios (0–1 cada):
    - clareza: texto não vazio, entre 1 e 300 palavras.
    - disclaimers: menciona que é apoio / não substitui médico / contexto clínico.
    - fidelidade_numerica: reflete a classe prevista (aceita negação/sinônimos).
    - contextualiza_features: cita ao menos uma feature (inglês ou português).
    - em_portugues: heurística — ausência de sinais de resposta em inglês.
    - sem_alucinacao_medica: não inventa prescrição/tratamento específico.

    O score total é normalizado pelo número de critérios **aplicáveis**
    (fidelidade e features só valem quando os valores esperados são informados).
    """
    text = text or ""
    palavras = len(text.split())

    def _contains_any(s: str, keys: list[str]) -> bool:
        return any(k.lower() in s.lower() for k in keys)

    clareza = 1.0 if 5 <= palavras <= 300 else 0.0
    disclaimers = 1.0 if _contains_any(
        text,
        ["apoio", "não substitui", "ferramenta", "avaliação clínica",
         "profissional de saúde", "em conjunto", "fatores clínicos",
         "confirmar o diagnóstico"],
    ) else 0.0

    # Fidelidade: busca por radical e aceita negação para a classe benigna
    fidelidade = None
    if expected_classe:
        fidelidade = 0.0
        txt = text.lower()
        radical = "malign" if expected_classe.startswith("malign") else "benign"
        if radical in txt:
            fidelidade = 1.0
        elif radical == "benign" and "malign" in txt and _contains_any(
            txt, ["não", "baixo", "baixa", "extremamente baixa", "não é elevada"]
        ):
            fidelidade = 1.0  # benigno expresso como "baixo risco / não maligno"

    # Features: aceita o nome em inglês ou traduções comuns em português
    feats = None
    if expected_features:
        feats = 0.0
        pt_map = {
            "radius": "raio", "texture": "textura", "perimeter": "perímetro",
            "area": "área", "smoothness": "suavidade", "rugosidade": "suavidade",
            "compactness": "compacidade", "concavity": "concavidade",
            "concave": "côncav", "symmetry": "simetria", "fractal": "fractal",
            "error": "erro", "mean": "média", "worst": "pior",
        }
        txt = text.lower()
        for feat in expected_features:
            base = feat.lower()
            if base in txt:
                feats = 1.0
                break
            for en, pt in pt_map.items():
                if en in base and pt in txt:
                    feats = 1.0
                    break
            if feats:
                break

    portugues = 1.0 if not _contains_any(text, ["the ", "this model ", "diagnosis is ", "benign tumor"]) else 0.0
    sem_aluc = 1.0 if not _contains_any(text, ["prescreva", "receite", "dose de", "recomendo tratamento com"]) else 0.0

    criterios = [clareza, disclaimers, portugues, sem_aluc]
    if fidelidade is not None:
        criterios.append(fidelidade)
    if feats is not None:
        criterios.append(feats)

    return {
        "clareza": clareza,
        "disclaimer": disclaimers,
        "fidelidade_numerica": fidelidade if fidelidade is not None else "n/a",
        "contextualiza_features": feats if feats is not None else "n/a",
        "em_portugues": portugues,
        "sem_alucinacao": sem_aluc,
        "score_total": round(sum(criterios) / len(criterios), 3),
        "num_palavras": palavras,
    }
