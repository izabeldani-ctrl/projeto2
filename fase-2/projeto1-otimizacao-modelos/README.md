# Projeto 1 — Otimização de Modelos de Diagnóstico (Fase 2)

Otimização dos modelos de diagnóstico da Fase 1 (câncer de mama) via
**Algoritmos Genéticos** + integração com **LLMs** para interpretação dos
resultados.

> 📋 Planejamento completo em [`../PLANO_EXECUCAO_FASE_2.md`](../PLANO_EXECUCAO_FASE_2.md)

## Status

- ✅ **Etapa 1 — Reuso da base da Fase 1** (concluída)
- ✅ **Etapa 2 — Algoritmo Genético** (concluída)
- ✅ **Etapa 3 — Experimentos** (concluída)
- ✅ **Etapa 4 — Comparativo otimizado × original** (concluída)
- ✅ **Etapa 5 — Monitoramento e logging** (concluída)
- ✅ **Etapa 6 — Integração com LLM** (concluída — Ollama local + llama3.2)
- ✅ **Etapa 7 — Testes automatizados** (concluída — 34 testes, 82% cobertura)
- ✅ **Etapa 8 — Relatório técnico** (concluída)
- ✅ **Etapa 10 — Entregáveis/repositório/notebook/checklist** (concluída)
- ⏳ **Etapa 9 — Vídeo de demonstração** (roteiro pronto; gravação depende do grupo)

## Estrutura

```text
projeto1-otimizacao-modelos/
├── src/
│   ├── data.py              # dataset + divisão treino/teste padronizada
│   ├── preprocessing.py     # pipeline numérica (imputação + escalonamento)
│   ├── evaluate.py          # métricas (accuracy, precision, recall, F1)
│   ├── ga/                  # ✅ Algoritmo Genético (Etapa 2)
│   │   ├── config.py            # GAConfig (população, taxas, operadores)
│   │   ├── operators.py         # seleção, crossover, mutação
│   │   ├── fitness.py           # accuracy média com CV estratificada 5-fold
│   │   ├── genetic_algorithm.py # loop principal com elitismo e estagnação
│   │   └── plotting.py          # gráficos de convergência
│   └── models/
│       ├── base_models.py   # modelos (LogReg, RF, SVC) + baselines da Fase 1
│       └── search_space.py  # espaço de busca de hiperparâmetros para o GA
│   └── monitoring/          # ✅ monitoramento (Etapa 5)
│       ├── logging_utils.py     # logging estruturado (console + arquivo)
│       └── tracker.py           # ExperimentTracker (CSV + JSON)
│   └── llm/                 # ✅ integração LLM (Etapa 6)
│       └── interpreter.py       # OllamaClient + prompts + avaliação de qualidade
├── run_baseline.py          # reproduz os baselines da Fase 1
├── run_ga.py                # executa o GA para um modelo (CLI)
├── run_experiments.py       # ✅ matriz de experimentos (Etapa 3)
├── compare_models.py        # ✅ comparativo baseline × GA no holdout (Etapa 4)
├── run_llm_demo.py          # ✅ demo de interpretação com LLM (Etapa 6)
├── notebooks/
│   └── demonstracao_fase2.ipynb  # ✅ demonstração ponta a ponta (Etapa 10)
├── reports/                    # ✅ relatório + saídas da LLM (Etapas 6/8)
│   ├── relatorio_fase2.md          # relatório técnico completo
│   ├── guia_video_fase2.md         # roteiro do vídeo (Etapa 9)
│   ├── checklist_entrega.md        # checklist final contra o enunciado
│   └── llm_output.md/json          # explicações geradas + avaliação de qualidade
├── logs/                    # ✅ logs estruturados (Etapa 5)
├── tests/                    # ✅ testes automatizados (Etapa 7)
│   ├── test_ga.py                # GA: encoding, operadores, fitness, convergência
│   ├── test_llm.py               # LLM: prompts, cliente (mock), avaliação
│   └── test_monitoring.py        # logging e tracker
├── results/
│   ├── metrics/             # baseline_*.csv, experiments_*.csv/json, comparativo_original_vs_ga.csv, ga_runs.csv/json
│   └── figures/             # convergence_*.png, experiments_<modelo>.png, comparativo_original_vs_ga.png
├── models/                  # baseline_<modelo>.joblib, ga_best_<modelo>.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

## Como executar

Reproduzir os baselines da Fase 1 (referência para o GA):

```powershell
python run_baseline.py
```

Otimizar um modelo com o Algoritmo Genético (Etapa 2):

```powershell
python run_ga.py --model svc --name exp1
python run_ga.py --model rf --pop 50 --gens 30 --mut 0.3 --selection roulette
python run_ga.py --model logreg --crossover blx_alpha --mutation uniform
```

Rodar a matriz de experimentos (Etapa 3):

```powershell
python run_experiments.py                          # todos os modelos
python run_experiments.py --models logreg svc      # selecionar modelos
python run_experiments.py --exps exp1 exp2         # selecionar experimentos
python run_experiments.py --scale 0.5              # reduzir pop/gen (modelos lentos)
```

Comparar otimizado × original no holdout (Etapa 4):

```powershell
python compare_models.py
```

Executar com monitoramento (Etapa 5) — logging estruturado + tracking:

```powershell
python run_ga.py --model svc --name exp1 --verbose   # log por geração (DEBUG)
python run_ga.py --model svc --name exp1 --log-file logs/ga_svc.log
```

O `run_experiments.py` também grava logs em `logs/experiments.log`.

Integração com LLM (Etapa 6) — exige Ollama rodando com o modelo baixado:

```powershell
# 1ª vez: instalar Ollama e baixar o modelo
ollama pull llama3.2
# Gerar explicações de diagnóstico com a LLM
python run_llm_demo.py
# Re-avaliar as explicações salvas (sem chamar a LLM de novo)
python run_llm_demo.py --re-evaluate
```

Rodar os testes:

```powershell
python -m pytest tests/ -q                                    # 34 testes
python -m pytest tests/ --cov=src --cov-report=term           # cobertura (82%)
```

## Resultados de referência (Etapa 1)

| Modelo | Accuracy | Precision | Recall | F1 | Best CV Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| SVC | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9736 |
| Regressão Logística | 0.9737 | 0.9726 | 0.9861 | 0.9793 | 0.9824 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9626 |

> Reprodução fiel dos resultados da Fase 1 — será a linha de base do comparativo
> com os modelos otimizados por Algoritmo Genético (Etapa 4).

## Resultados dos experimentos (Etapa 3)

Melhor fitness = accuracy média com validação cruzada 5-fold (mesma da Fase 1).
Arquivos: `results/metrics/experiments_summary.csv` e `experiments_details.json`.

| Modelo | Exp | Pop | Gens | Mut | Seleção | Melhor fitness | Baseline |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| LogReg | exp1 | 10 | 15 | 0.1 | torneio | 0.9824 | 0.9824 |
| LogReg | exp2 | 16 | 15 | 0.1 | torneio | **0.9846** | 0.9824 |
| LogReg | exp3 | 10 | 20 | 0.3 | roleta | **0.9846** | 0.9824 |
| LogReg | exp4 | 16 | 20 | 0.15 | torneio | **0.9846** | 0.9824 |
| RF | exp1 | 5 | 8 | 0.1 | torneio | 0.9604 | 0.9626 |
| RF | exp2 | 8 | 8 | 0.1 | torneio | 0.9626 | 0.9626 |
| RF | exp3 | 5 | 10 | 0.3 | roleta | 0.9648 | 0.9626 |
| RF | exp4 | 8 | 10 | 0.15 | torneio | **0.9670** | 0.9626 |
| SVC | exp1 | 10 | 15 | 0.1 | torneio | 0.9626 | 0.9736 |
| SVC | exp2 | 16 | 15 | 0.1 | torneio | **0.9802** | 0.9736 |
| SVC | exp3 | 10 | 20 | 0.3 | roleta | **0.9802** | 0.9736 |
| SVC | exp4 | 16 | 20 | 0.15 | torneio | **0.9758** | 0.9736 |

- O GA igualou ou superou o baseline na maioria dos experimentos.
- Maiores ganhos: **SVC** (exp2/exp3, +0.66 pp) e **LogReg** (exp2–4, +0.22 pp).
- Para o RF foi usada escala 0.5 (custo computacional) e `n_estimators` limitado
  a (20, 200); o exp4 atingiu 0.9670 (acima do baseline).
- Figuras de convergência: `results/figures/convergence_<modelo>_<exp>.png` e
  `experiments_<modelo>.png` (todas as curvas comparadas).

## Comparativo: baseline × otimizado no holdout (Etapa 4)

Melhores resultados de **accuracy no holdout** (test_size=0.2, random_state=42) —
`results/metrics/comparativo_original_vs_ga.csv`:

| Modelo | Baseline (GridSearch) | Melhor GA | Ganho |
| --- | ---: | ---: | ---: |
| LogReg | 0.9737 | **0.9825** (exp1–3) | **+0.88 pp** |
| RF | 0.9561 | 0.9561 (exp2–4) | 0 |
| SVC | 0.9825 | 0.9825 (exp2–4) | 0 |

**Leitura crítica:**
- **LogReg**: o GA encontrou configuração que **generaliza melhor no holdout**
  (+0.88 pp), explorando regiões fora do grid da Fase 1 (C em escala fina).
- **SVC/RF**: o GA empatou com o baseline (já muito bom) — confirma robustez.
- **Estabilidade**: experimentos com população pequena (ex.: SVC exp1) ficaram
  piores no holdout apesar da boa fitness de CV — otimizar a fitness de CV nem
  sempre maximiza a generalização.

## Monitoramento e logging (Etapa 5)

O projeto registra cada execução de forma estruturada, permitindo auditoria e
comparação de experimentos:

- **Logging** (`src/monitoring/logging_utils.py`): formato consistente
  `timestamp | nível | módulo | mensagem`, com saída em console e/ou arquivo
  (`logs/`). O GA loga início/fim em `INFO` e, com `--verbose`, cada geração em
  `DEBUG` (melhor fitness, média, estagnação).
- **Tracking** (`src/monitoring/tracker.py`): `ExperimentTracker` persiste cada
  execução em `results/metrics/ga_runs.csv` (resumo) e `ga_runs.json` (detalhe
  com config, parâmetros e métricas) — histórico acumulado entre execuções.

### Arquitetura da solução

```mermaid
flowchart LR
    A[Dados<br/>Breast Cancer<br/>Wisconsin] --> B[Divisão<br/>treino/teste<br/>padronizada]
    B --> C[Baselines Fase 1<br/>GridSearchCV]
    B --> D[Algoritmo Genético<br/>src/ga/]
    D --> E[Fitness<br/>accuracy CV 5-fold]
    E --> F[Experimentos<br/>4 configs × 3 modelos]
    F --> G[Comparativo<br/>otimizado × original<br/>no holdout]
    G --> H[Melhores modelos<br/>ga_best_*.joblib]
    D -.-> I[Monitoramento<br/>logging + tracker]
    F -.-> I
    I --> J[logs/ + ga_runs.csv|json]
```

> Escalabilidade: o tracking em CSV/JSON é a camada local; é possível evoluir
> para MLflow/W&B ou nuvem (IaC) como pontuação extra — os artefatos já são
> versionáveis e reproduzíveis (sementes fixas).

## Integração com LLM (Etapa 6)

Gera **explicações em linguagem natural** dos diagnósticos via **Ollama local +
llama3.2 (3B)** — sem custo e sem nuvem (alternativas: Groq free / OpenAI pago).

- **Prompt engineering** (`src/llm/interpreter.py`): prompt de sistema com papel
  clínico e regras (não inventar dados, não substituir o médico, português,
  conciso); contexto estruturado com classe, probabilidade e as features mais
  relevantes (nomes traduzidos para pt); cliente HTTP com retry/fallback.
- **Saídas** (`run_llm_demo.py` → `reports/llm_output.md|json`): explicação de 3
  casos (benigno, maligno e caso-limite) + interpretação agregada do modelo.
- **Avaliação de qualidade** (`evaluate_explanation`): checklist com score
  normalizado (clareza, disclaimer, fidelidade numérica, features em pt,
  português, sem alucinação).

### Qualidade das explicações geradas (llama3.2)

| Saída | Score |
| --- | ---: |
| Caso benigno (maior confiança) | 1.0 |
| Caso maligno (maior confiança) | 1.0 |
| Caso-limite (maior incerteza) | 1.0 |
| Interpretação agregada do modelo | 1.0 |

Exemplo (caso-limite, probabilidade 47,3% de malignidade):

> "…há uma probabilidade de 47,3% de que o tumor seja maligno… a probabilidade
> não é elevada… deve ser considerada em conjunto com outros fatores clínicos e
> de imagem."

> Nota: o avaliador automático foi calibrado para reconhecer negações ("não é
> maligno", "baixo risco") e sinônimos em português ("avaliação clínica"); uma
> revisão manual confirmou a fidelidade e a ausência de alucinação.

## Espaço de busca (para o GA — Etapa 2)

| Modelo | Hiperparâmetros | Tipo |
| --- | --- | --- |
| LogReg | `C` | float (log) |
| | `class_weight` | categórico |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` | inteiro |
| SVC | `C`, `gamma` | float (log) |
| | `kernel` | categórico |

Genes representados normalizados em `[0, 1]`; `decode_genes()` converte para os
parâmetros do scikit-learn.
