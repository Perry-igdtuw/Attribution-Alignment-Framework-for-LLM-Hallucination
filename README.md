# Multi-Dimensional Evaluation of Explanation Faithfulness and Hallucination in LLMs

> **Research Project** | Explainable AI · LLM Hallucination Detection · Causal Perturbation Analysis

---

## Overview

This project proposes and evaluates the **Faithfulness-Hallucination Index (FHI)** — a novel composite metric for detecting hallucinations in Large Language Models by analyzing the alignment between model explanations and internal attribution signals.

### Core Hypothesis
Hallucinations arise when a model's explanation is **causally disconnected** from its actual reasoning process. FHI captures this disconnect through four complementary dimensions:

| Metric | Description | Role in FHI |
|---|---|---|
| **AAS** | Attribution Alignment Score | Structural overlap between explanation and attribution |
| **CIS** | Causal Impact Score | Output change when explanation tokens are removed |
| **ESS** | Explanation Stability Score | Consistency across multiple generation runs |
| **HCG** | Hallucination Confidence Gap | Confidence minus factual correctness |
| **FHI** | Faithfulness-Hallucination Index | `w1·AAS + w2·CIS + w3·ESS - w4·HCG` |

---

## Setup

### 1. Python Environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. HuggingFace Login (for Gemma)

Gemma requires accepting the license at https://huggingface.co/google/gemma-2b-it

```bash
pip install huggingface_hub
huggingface-cli login           # Enter your HF token
```

### 3. (Optional) Ollama Setup

For fast inference without XAI:
```bash
brew install ollama             # macOS
ollama serve                    # Start server
ollama pull gemma:2b            # Download model
```

### 4. Configure

Edit `config/model_config.yaml`:
- Set `backend: "huggingface"` (for XAI) or `"ollama"` (for fast inference)
- Set `device: "cuda"` / `"mps"` / `"cpu"` based on your hardware
- Set `hf_model_id: "google/gemma-2b-it"` or `"google/gemma-7b-it"`

---

## Running

```bash
# Quick test (5 samples, validates the pipeline works)
python experiments/run_pipeline.py --test

# Full run on all datasets
python experiments/run_pipeline.py

# Specific dataset only
python experiments/run_pipeline.py --dataset trivia_qa

# Use Ollama backend (fast, no XAI internals)
python experiments/run_pipeline.py --backend ollama
```

---

## Project Structure

```
research/
├── config/                 # YAML configuration (model, experiment, weights)
├── src/
│   ├── model/              # LLM client, token extractor, prompt templates
│   ├── xai/                # Attention, gradient, SHAP attribution methods
│   ├── perturbation/       # Token masking, output comparison
│   ├── metrics/            # AAS, CIS, ESS, HCG, FHI computation
│   ├── evaluation/         # Baselines, evaluator, statistical tests
│   ├── data/               # Dataset loading, adversarial generation
│   └── visualization/      # All plot generation scripts
├── experiments/            # Pipeline entry points
├── results/                # Raw JSON + processed CSVs + figures
├── notebooks/              # Jupyter notebooks for exploration
├── paper/                  # LaTeX paper
└── tests/                  # Unit tests
```

---

## FHI Formula

```
FHI = clip(w1·AAS + w2·CIS + w3·ESS - w4·HCG, 0, 1)

Default weights:
  w1 = 0.30  (Attribution Alignment)
  w2 = 0.35  (Causal Impact — highest: causal evidence is most reliable)
  w3 = 0.20  (Explanation Stability)
  w4 = 0.15  (Confidence Gap — subtracted: penalizes overconfident wrong answers)

Threshold: FHI < 0.5 → hallucination detected
```

---

## Datasets

| Dataset | Type | HF Hub ID |
|---|---|---|
| TriviaQA | Factual QA | `trivia_qa` |
| HaluEval | Hallucination benchmark | `pminervini/HaluEval` |
| MuSiQue | Multi-hop reasoning | `musique` |

---

## Citation

If you use this work, cite:

```bibtex
@article{fhi2025,
  title   = {A Multi-Dimensional Evaluation of Explanation Faithfulness and 
             Hallucination in LLMs using Attribution Alignment and 
             Causal Perturbation Analysis},
  author  = {[Author]},
  year    = {2025},
}
```
