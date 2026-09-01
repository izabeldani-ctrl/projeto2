"""Testes do módulo de integração com LLM (Etapa 6/7).

Cobre tradução de features, builders de prompt, cliente Ollama (com mock da
API HTTP) e o checklist de avaliação de qualidade.
"""
from __future__ import annotations

import json
from unittest import mock

from src.llm.interpreter import (
    OllamaClient,
    build_individual_prompt,
    build_model_summary_prompt,
    evaluate_explanation,
    translate_feature_name,
)


# ---------------------------------------------------------------------------
# Tradução de features
# ---------------------------------------------------------------------------

def test_translate_feature_name_basic() -> None:
    assert translate_feature_name("worst area") == "pior área"
    assert translate_feature_name("mean radius") == "média raio"
    assert translate_feature_name("texture error") == "textura erro"


def test_translate_feature_name_multiword() -> None:
    assert translate_feature_name("mean concave points") == "média pontos côncavos"
    assert translate_feature_name("worst fractal dimension") == "pior dimensão fractal"


# ---------------------------------------------------------------------------
# Builders de prompt
# ---------------------------------------------------------------------------

def test_build_individual_prompt_contains_values() -> None:
    p = build_individual_prompt(
        model_name="LogReg",
        prediction=1,
        proba_maligna=0.9,
        top_features=[("worst area", 123.4)],
    )
    assert "MALIGNO" in p
    assert "90.0%" in p
    assert "pior área = 123.400" in p


def test_build_model_summary_prompt_contains_metrics() -> None:
    p = build_model_summary_prompt(
        model_name="LogReg",
        metrics={"accuracy": 0.98, "f1": 0.97},
        best_params={"C": 0.1},
        gains={"accuracy": 0.01},
    )
    assert "accuracy=0.9800" in p
    assert "0.0100" in p


# ---------------------------------------------------------------------------
# OllamaClient (mock da API HTTP)
# ---------------------------------------------------------------------------

class _FakeResponse:
    status = 200

    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> bytes:
        return self._text.encode("utf-8")

    def __enter__(self):  # urlopen é usado como context manager
        return self

    def __exit__(self, *args):  # noqa: ANN002
        return False


def test_generate_returns_response() -> None:
    client = OllamaClient()
    with mock.patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps({"response": "olá mundo"})),
    ):
        assert client.generate("prompt") == "olá mundo"


def test_generate_sends_correct_payload() -> None:
    client = OllamaClient(model="llama3.2")
    captured: dict = {}

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(json.dumps({"response": "ok"}))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.generate("prompt", system="sys", temperature=0.1)

    assert captured["url"].endswith("/api/generate")
    assert captured["body"]["model"] == "llama3.2"
    assert captured["body"]["system"] == "sys"
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["temperature"] == 0.1


def test_generate_retries_then_fallback() -> None:
    client = OllamaClient()
    with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timeout")):
        result = client.generate("prompt", retries=1, fallback="FALLBACK")
    assert result == "FALLBACK"


def test_is_available_true() -> None:
    client = OllamaClient()
    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse("{}")):
        assert client.is_available() is True


def test_is_available_false() -> None:
    client = OllamaClient()
    with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
        assert client.is_available() is False


# ---------------------------------------------------------------------------
# Avaliação de qualidade
# ---------------------------------------------------------------------------

def test_evaluate_benigno_negacao() -> None:
    """Benigno expresso como 'baixo risco de malignidade' deve contar como fiel."""
    texto = "Probabilidade baixa de malignidade. O modelo é uma ferramenta de apoio e não substitui o médico."
    aval = evaluate_explanation(texto, expected_classe="benigno", expected_features=[])
    assert aval["fidelidade_numerica"] == 1.0
    assert aval["score_total"] == 1.0


def test_evaluate_detecta_alucinacao() -> None:
    texto = "Recomendo prescreva a dose de X. Apoio e não substitui o médico."
    aval = evaluate_explanation(texto, expected_classe="maligno", expected_features=[])
    assert aval["sem_alucinacao"] == 0.0


def test_evaluate_agregado_sem_penalidade() -> None:
    """Sem classe/features esperadas, o score não é penalizado por critérios n/a."""
    texto = "O modelo é uma ferramenta de apoio e não substitui a avaliação clínica."
    aval = evaluate_explanation(texto, expected_features=[])
    assert aval["fidelidade_numerica"] == "n/a"
    assert aval["score_total"] == 1.0


def test_evaluate_features_em_portugues() -> None:
    texto = "A textura e a simetria do exame indicam maior risco de malignidade. Ferramenta de apoio."
    aval = evaluate_explanation(
        texto,
        expected_classe="maligno",
        expected_features=["worst texture", "pior textura"],
    )
    assert aval["contextualiza_features"] == 1.0
    assert aval["fidelidade_numerica"] == 1.0
