# Faithfulness-Hallucination Index (FHI)

**A Multi-Dimensional Evaluation of Explanation Faithfulness and Hallucination in Large Language Models using Attribution Alignment and Causal Perturbation Analysis**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Paper%20Status-Under%20Review-yellow)
![Model](https://img.shields.io/badge/LLM-Gemma--2B--it-purple)

---

## Overview

Large Language Models (LLMs) frequently produce fluent but factually incorrect outputs — a phenomenon known as *hallucination*. While Chain-of-Thought (CoT) prompting can improve reasoning, self-generated explanations often act as post-hoc rationalizations that do not faithfully reflect the model's internal reasoning process.

This project introduces the **Faithfulness-Hallucination Index (FHI)**: a composite evaluation framework that detects hallucination by measuring *how faithfully a model's explanation represents its actual reasoning*, using attribution alignment and causal perturbation analysis.

> **Core Hypothesis:** Hallucinations arise from misalignment between a model's internal reasoning (captured via XAI) and its surface-level explanation. Faithfulness is multi-dimensional — combining attribution analysis with causal perturbation yields a stronger hallucination signal than any single metric.

---

## System Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  LLM Client  │────▶│  XAI Attribution│────▶│  Causal          │
│  (Gemma-2B)  │     │  • Attention    │     │  Perturbation    │
│              │     │  • Gradients    │     │  • Mask/Delete   │
│  Answer +    │     │  • SHAP         │     │  • Replace       │
│  Explanation │     │                 │     │  • Compare       │
└──────────────┘     └─────────────────┘     └──────────────────┘
       │                     │                        │
       ▼                     ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Metric Layer                             │
│  AAS (Attribution Alignment)  ─┐                               │
│  CIS (Causal Impact)          ─┼──▶  FHI = w₁·AAS + w₂·CIS   │
│  ESS (Explanation Stability)  ─┤         + w₃·ESS − w₄·HCG   │
│  HCG (Confidence Gap)         ─┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Metrics

| Metric | Full Name | What It Measures | Weight |
|--------|-----------|------------------|--------|
| **AAS** | Attribution Alignment Score | Overlap between explanation tokens and XAI-highlighted tokens | 0.25 |
| **CIS** | Causal Impact Score | Answer shift when explanation-referenced tokens are removed | 0.35 |
| **ESS** | Explanation Stability Score | Consistency of explanations across multiple generation runs | 0.25 |
| **HCG** | Hallucination Confidence Gap | Penalty for high-confidence incorrect answers | 0.15 |

The FHI composite score:

```
FHI = 0.25 · AAS + 0.35 · CIS + 0.25 · ESS − 0.15 · HCG
```

---

## Results — Gemma-2B-it on HaluEval

```
┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Metric ┃  Mean  ┃  Std   ┃
┡━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ FHI    │ 0.3184 │ 0.0525 │
│ AAS    │ 0.0956 │ 0.0751 │
│ CIS    │ 0.4876 │ 0.1235 │
│ ESS    │ 0.7397 │ 0.1957 │
│ HCG    │ 0.1925 │ 0.1388 │
└────────┴────────┴────────┘

FHI Classification Accuracy: 60%  |  F1 (Hallucination): 0.75
```

**Key findings:**

- **Low AAS (0.096)** — Gemma's explanations reference different tokens than those highlighted by its own attention mechanism, providing strong evidence of post-hoc rationalization rather than faithful explanation.
- **High ESS (0.740)** — The model produces *consistent* but unfaithful explanations: it reliably generates the same flawed reasoning, which is a systematic rather than random failure.
- **CIS ≈ 0.49** — Removing explanation-grounded tokens causes only partial answer shifts, indicating mixed causal grounding and motivating the need for a composite metric.

---

## Project Structure

```
research/
├── config/                         # YAML configuration files
│   ├── model_config.yaml           # Backend, model ID, generation params
│   ├── experiment_config.yaml      # Dataset paths, attribution methods
│   └── metric_weights.yaml         # FHI weight tuning (w1–w4)
│
├── src/
│   ├── model/                      # LLM inference layer
│   │   ├── llm_client.py           # Unified HuggingFace + Ollama client
│   │   ├── token_extractor.py      # Log-prob, attention, gradient extraction
│   │   └── prompt_templates.py     # CoT, self-explanation, direct prompts
│   ├── xai/                        # Explainability methods
│   │   ├── attention_attribution.py  # Attention rollout (Abnar & Zuidema 2020)
│   │   ├── gradient_attribution.py   # Integrated Gradients via Captum
│   │   ├── shap_approximator.py      # Coalition-sampling SHAP
│   │   └── attribution_utils.py      # Normalization & token alignment
│   ├── perturbation/               # Causal perturbation engine
│   │   ├── token_masker.py         # Mask, delete, replace strategies
│   │   ├── output_comparator.py    # JS divergence, semantic distance, ROUGE-L
│   │   └── causal_engine.py        # Orchestrates full causal impact analysis
│   ├── metrics/                    # FHI component metrics
│   │   ├── aas.py                  # Attribution Alignment Score
│   │   ├── cis.py                  # Causal Impact Score
│   │   ├── ess.py                  # Explanation Stability Score
│   │   ├── hcg.py                  # Hallucination Confidence Gap
│   │   └── fhi.py                  # Composite FHI + weight optimizer
│   ├── data/                       # Dataset loading & adversarial generation
│   │   ├── dataset_loader.py       # TriviaQA, HaluEval, MuSiQue loaders
│   │   └── adversarial_generator.py  # False premise injection
│   ├── evaluation/                 # Baselines & statistical evaluation
│   │   ├── baselines.py            # LogProb thresholding, Self-Consistency
│   │   └── evaluator.py            # Precision / Recall / F1 / AUC-ROC
│   └── visualization/              # Publication-quality plotting
│       └── plot_metrics.py         # KDE, scatter, ROC, radar charts
│
├── experiments/
│   └── run_pipeline.py             # Main entry point with MLflow tracking
│
├── dashboard/                      # Interactive web dashboard
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── paper/                          # LaTeX paper (IEEE format)
│   ├── main.tex
│   ├── references.bib
│   └── sections/
│       ├── 1_introduction.tex
│       ├── 2_related_work.tex
│       ├── 3_methodology.tex
│       ├── 4_experimental_setup.tex
│       ├── 5_results.tex
│       └── 6_conclusion.tex
│
├── notebooks/                      # Jupyter exploration notebooks
├── tests/                          # Unit & integration tests
├── results/                        # Experiment outputs (gitignored)
├── EXPLANATION.md                  # Detailed pipeline walkthrough
└── requirements.txt                # Python dependencies
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- HuggingFace account with access to [Gemma-2B-it](https://huggingface.co/google/gemma-2b-it)
- 8 GB RAM minimum (CPU-compatible; GPU/MPS optional)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/fhi-llm-evaluation.git
cd fhi-llm-evaluation

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Authenticate with HuggingFace (required for Gemma access)
hf auth login
```

### Running Experiments

```bash
# Quick test run (5 samples, validates the full pipeline)
python experiments/run_pipeline.py --test

# Full run on HaluEval
python experiments/run_pipeline.py --dataset halueval

# Full run on TriviaQA
python experiments/run_pipeline.py --dataset triviaqa

# View MLflow experiment tracking
mlflow ui
# Open http://localhost:5000
```

### Launch the Dashboard

```bash
python3 -m http.server 8080 -d dashboard/
# Open http://localhost:8080
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Google Gemma-2B-it (HuggingFace Transformers) |
| XAI | Captum (Integrated Gradients), Custom Attention Rollout |
| Semantic Similarity | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Experiment Tracking | MLflow |
| Datasets | HuggingFace Datasets (TriviaQA, HaluEval, MuSiQue) |
| Visualization | Matplotlib, Seaborn, Chart.js |
| Compute | CPU-compatible (GPU / Apple MPS optional) |

---

## Paper

The full IEEE-format paper is in [`paper/`](./paper/). It is written in LaTeX and structured as a conference submission. The compiled PDF is not tracked — build it locally with:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex && pdflatex main.tex
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{fhi2026,
  title   = {A Multi-Dimensional Evaluation of Explanation Faithfulness and
             Hallucination in Large Language Models using Attribution Alignment
             and Causal Perturbation Analysis},
  author  = {Singh, Perry},
  year    = {2026},
  note    = {Final Year Research Paper}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
