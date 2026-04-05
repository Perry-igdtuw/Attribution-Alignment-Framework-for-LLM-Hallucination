# Multi-Dimensional Evaluation of Explanation Faithfulness and Hallucination in LLMs

## Project Summary

This project investigates the causal relationship between **explanation faithfulness** and **hallucination** in Large Language Models. We propose a novel composite metric — the **Faithfulness-Hallucination Index (FHI)** — that integrates attribution alignment, causal perturbation, explanation stability, and confidence gap analysis. The system is evaluated on factual QA, adversarial, and multi-hop reasoning benchmarks.

**Target Venues**: EMNLP, ACL Findings, EACL, or SCOPUS-indexed journals (e.g., *Expert Systems With Applications*, *Information Fusion*)

---

## Phase 1: Architecture, Folder Structure & Tech Stack

---

### 1.1 System Architecture Overview

The pipeline is composed of **6 independent, composable modules**:

```
┌──────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                          │
│   Question (Q) ──→ Prompt Template ──→ Tokenized Input       │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                         LLM MODULE                           │
│   Gemma (via Ollama API / HuggingFace transformers)          │
│   Outputs: Answer (A), Log-probabilities, Attention Weights  │
└───────┬───────────────────────────────┬──────────────────────┘
        │                               │
┌───────▼──────────┐         ┌──────────▼────────────┐
│ EXPLANATION GEN  │         │   XAI ATTRIBUTION      │
│  Chain-of-Thought│         │   • Attention rollout  │
│  or Self-explain │         │   • SHAP (text approx) │
│  Prompt method   │         │   • Gradient×Input     │
└───────┬──────────┘         └──────────┬─────────────┘
        │                               │
┌───────▼───────────────────────────────▼──────────────────────┐
│                    ALIGNMENT LAYER                           │
│   AAS: Overlap(Explanation Tokens, Attribution Top-K Tokens) │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│               CAUSAL PERTURBATION MODULE                     │
│   Strategy: Mask / Replace / Delete explanation tokens       │
│   Measure: JS Divergence / ROUGE / Semantic Similarity shift │
│   Output: CIS (Causal Impact Score)                          │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                    METRIC AGGREGATION                        │
│   • AAS  – Attribution Alignment Score                       │
│   • CIS  – Causal Impact Score                               │
│   • ESS  – Explanation Stability Score                       │
│   • HCG  – Hallucination Confidence Gap                      │
│   → FHI = w1·AAS + w2·CIS + w3·ESS - w4·HCG                 │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                  EVALUATION & COMPARISON                     │
│   Baselines: Log-prob | Self-Consistency | RAG-Verify | FHI  │
│   Metrics: Accuracy, Precision, Recall, F1, AUC-ROC          │
└──────────────────────────────────────────────────────────────┘
```

---

### 1.2 Folder Structure

```
research/
│
├── README.md                          # Project overview + setup guide
├── requirements.txt                   # All Python dependencies
├── config/
│   ├── model_config.yaml              # Model name, temperature, max_tokens
│   ├── experiment_config.yaml         # Dataset paths, run settings
│   └── metric_weights.yaml            # w1–w4 for FHI, tunable
│
├── src/
│   ├── __init__.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── llm_client.py              # Unified LLM API (Ollama / HuggingFace)
│   │   ├── token_extractor.py         # Extract log-probs, attention weights
│   │   └── prompt_templates.py        # CoT prompts, self-explanation prompts
│   │
│   ├── xai/
│   │   ├── __init__.py
│   │   ├── attention_attribution.py   # Attention rollout implementation
│   │   ├── gradient_attribution.py    # Gradient × Input (transformers-based)
│   │   ├── shap_approximator.py       # SHAP via LIME-style text perturbation
│   │   └── attribution_utils.py       # Normalization, token alignment helpers
│   │
│   ├── perturbation/
│   │   ├── __init__.py
│   │   ├── token_masker.py            # Mask / replace / delete strategies
│   │   ├── output_comparator.py       # JS divergence, semantic sim measurement
│   │   └── causal_engine.py           # Orchestrates perturbation experiments
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── aas.py                     # Attribution Alignment Score
│   │   ├── cis.py                     # Causal Impact Score
│   │   ├── ess.py                     # Explanation Stability Score
│   │   ├── hcg.py                     # Hallucination Confidence Gap
│   │   └── fhi.py                     # Faithfulness-Hallucination Index
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── baselines.py               # Log-prob, self-consistency, RAG-verify
│   │   ├── evaluator.py               # Main evaluation orchestrator
│   │   └── statistical_tests.py       # t-test, Mann-Whitney, effect sizes
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py          # Load TriviaQA, HaluEval, MuSiQue
│   │   ├── adversarial_generator.py   # Create hallucination-inducing samples
│   │   └── preprocessor.py            # Tokenization, formatting
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── correlation_plots.py        # AAS vs FHI correlation heatmaps
│       ├── ablation_plots.py           # Ablation study bar charts
│       ├── metric_comparison.py        # Baseline vs FHI radar/bar plots
│       └── attention_heatmap.py        # Token-level attention visualization
│
├── experiments/
│   ├── run_pipeline.py                # Main entry point: full pipeline run
│   ├── run_ablation.py                # Ablation: remove each metric component
│   ├── run_baselines.py               # Run all baseline comparisons
│   └── run_visualization.py           # Generate all paper figures
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   ├── 03_metric_analysis.ipynb
│   └── 04_results_visualization.ipynb
│
├── results/
│   ├── raw/                           # JSON outputs per experiment run
│   ├── processed/                     # Aggregated CSVs
│   └── figures/                       # All generated plots (PDF + PNG)
│
├── paper/
│   ├── main.tex                       # LaTeX paper
│   ├── refs.bib                       # Bibliography
│   ├── sections/
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── related_work.tex
│   │   ├── methodology.tex
│   │   ├── experiments.tex
│   │   ├── results.tex
│   │   └── conclusion.tex
│   └── figures/                       # Symlink or copy of results/figures
│
└── tests/
    ├── test_metrics.py                # Unit tests for AAS, CIS, ESS, HCG, FHI
    ├── test_perturbation.py           # Unit tests for token masking strategies
    └── test_attribution.py            # Unit tests for XAI methods
```

---

### 1.3 Tech Stack Decisions (Justified)

| Component | Choice | Justification |
|---|---|---|
| **LLM** | Gemma-2B / Gemma-7B via Ollama | Local inference, no API cost, reproducible, supports log-prob extraction |
| **Transformers** | HuggingFace `transformers` | Needed for attention weight + gradient access (Ollama doesn't expose internals) |
| **XAI** | Custom attention rollout + `captum` | `captum` is PyTorch's official XAI library; more reliable than raw SHAP for text |
| **SHAP** | `shap` with `Explainer` for text | Widely cited in XAI literature; reviewers recognize it |
| **Semantic Similarity** | `sentence-transformers` (all-MiniLM-L6-v2) | Fast, accurate, model-agnostic semantic comparison |
| **NLP Utilities** | `nltk`, `rouge-score`, `bert-score` | Standard metrics used in NLG evaluation papers |
| **Data** | HuggingFace `datasets` | Reproducible dataset loading with versioning |
| **Config** | `pydantic` + YAML | Type-safe configuration, easy to document for paper |
| **Experiment Tracking** | `mlflow` or `wandb` | Required for reproducibility claims in top venues |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` | Publication-quality figures |
| **Statistical Testing** | `scipy.stats` | t-tests, Mann-Whitney U — reviewers expect significance testing |
| **Paper** | LaTeX + `acl_natbib` style | ACL/EMNLP standard |

---

### 1.4 Module APIs (Interface Contracts)

#### `LLMClient` → standardized output
```python
@dataclass
class LLMResponse:
    answer: str                        # Generated text
    explanation: str                   # CoT or self-explanation
    token_log_probs: List[float]       # Per-token log probabilities
    attention_weights: Optional[List[np.ndarray]]  # Layer × Head × Seq × Seq
    tokens: List[str]                  # Token strings
    confidence: float                  # Mean prob of answer tokens
```

#### `AttributionResult` → standardized attribution output
```python
@dataclass
class AttributionResult:
    tokens: List[str]
    scores: np.ndarray                 # Shape: (n_tokens,), normalized [0,1]
    method: str                        # "attention" | "gradient" | "shap"
    top_k_tokens: List[str]            # Top-K most important tokens
```

#### `PerturbationResult` → causal experiment output
```python
@dataclass
class PerturbationResult:
    original_output: str
    perturbed_output: str
    strategy: str                      # "mask" | "delete" | "replace"
    js_divergence: float
    semantic_shift: float              # Cosine distance between embeddings
    rouge_delta: float
```

#### `MetricBundle` → all metrics for one sample
```python
@dataclass
class MetricBundle:
    sample_id: str
    aas: float                         # [0, 1]
    cis: float                         # [0, 1]
    ess: float                         # [0, 1]
    hcg: float                         # [0, 1] — to be subtracted
    fhi: float                         # [0, 1] — final score
    is_hallucination: bool             # Ground truth label
    predicted_hallucination: bool      # FHI threshold prediction
```

---

### 1.5 Dual-Mode LLM Strategy (Critical Design Decision)

> [!IMPORTANT]
> **You need TWO modes of LLM access** — this is not optional, it's architecturally necessary:
>
> - **Ollama mode**: Fast inference for answer + explanation generation (production-like)
> - **HuggingFace mode**: Required for attention weights and gradient-based attribution (internals access)
>
> The `llm_client.py` will abstract both behind a unified interface. Experiments will use HF mode by default.

---

## Open Questions for User Confirmation

> [!IMPORTANT]
> **Q1 — Model Choice**: Do you prefer Gemma-2B (faster, fits 8GB RAM) or Gemma-7B (more capable, needs ~16GB)? Or should we support both with a config flag?

> [!IMPORTANT]
> **Q2 — Compute Environment**: Do you have a GPU available locally? If not, we should design for CPU-compatible execution with smaller batch sizes.

> [!WARNING]
> **Q3 — SHAP vs Captum**: SHAP text explanations are computationally expensive (~5–10 min/sample). `captum` (Integrated Gradients) is faster and equally well-cited. Should we implement both or prioritize `captum`?

> [!NOTE]
> **Q4 — Dataset Scale**: For a publication, we need at least 500–1000 samples per dataset split. Do you have time/compute constraints I should factor in?

---

## Verification Plan

### After Phase 1 (this step)
- User reviews and approves architecture
- We scaffold the folder structure with stub files

### After Phase 2 (Model Setup)
- Run `python experiments/run_pipeline.py --test` on 5 samples
- Confirm LLMResponse object is populated correctly

### Final Verification
- Full pipeline run on 3 datasets
- Ablation study confirms each metric contributes positively to FHI
- Statistical significance tests pass (p < 0.05)
- All figures generated and paper draft complete
