# Attribution Alignment Framework for LLM Hallucination

**A Multi-Dimensional Evaluation of Explanation Faithfulness and Hallucination in Large Language Models using Attribution Alignment and Causal Perturbation Analysis**

---

## Abstract

Large Language Models (LLMs) often generate fluent but factually incorrect outputs — hallucinations. While self-generated explanations (Chain-of-Thought) can improve reasoning, they can also act as post-hoc rationalizations that don't faithfully represent the model's internal reasoning. This project introduces the **Faithfulness-Hallucination Index (FHI)**, a multi-dimensional evaluation framework that correlates explanation unfaithfulness with factual hallucination.

## Core Hypothesis

> Hallucinations arise from misalignment between model reasoning and generated explanations. Faithfulness is multi-dimensional and cannot be captured by a single metric. Combining attribution + causal perturbation provides better hallucination detection.

---

## Architecture

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
│                    Metric Computation                          │
│  AAS (Attribution Alignment)  ─┐                               │
│  CIS (Causal Impact)          ─┼──▶  FHI = w₁·AAS + w₂·CIS   │
│  ESS (Explanation Stability)  ─┤         + w₃·ESS - w₄·HCG   │
│  HCG (Confidence Gap)         ─┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Metrics

| Metric | Full Name | What it Measures | Weight |
|--------|-----------|------------------|--------|
| **AAS** | Attribution Alignment Score | Overlap between explanation tokens and XAI-highlighted tokens | 0.25 |
| **CIS** | Causal Impact Score | How much the answer changes when explanation words are removed | 0.35 |
| **ESS** | Explanation Stability Score | Consistency of explanations across multiple runs | 0.25 |
| **HCG** | Hallucination Confidence Gap | Penalty for high confidence on incorrect answers | 0.15 |

## Results (Gemma-2B-it on HaluEval)

```
┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Metric ┃ Mean   ┃ Std    ┃
┡━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ FHI    │ 0.3184 │ 0.0525 │
│ AAS    │ 0.0956 │ 0.0751 │
│ CIS    │ 0.4876 │ 0.1235 │
│ ESS    │ 0.7397 │ 0.1957 │
│ HCG    │ 0.1925 │ 0.1388 │
└────────┴────────┴────────┘
FHI Classification Accuracy: 60% | F1 (Hallucination): 0.75
```

**Key Findings:**
- **Low AAS (0.10)**: Gemma's explanations reference different tokens than what its attention mechanism used internally — strong evidence of post-hoc rationalization
- **High ESS (0.74)**: The model gives consistent but unfaithful explanations — it reliably hallucinating the same reasoning
- **CIS near 0.49**: Removing explanation-highlighted tokens partially shifts the answer, showing mixed causal grounding

## Project Structure

```
research/
├── config/                     # YAML configuration files
│   ├── model_config.yaml       # Backend, model ID, generation params
│   ├── experiment_config.yaml  # Dataset paths, attribution methods
│   └── metric_weights.yaml     # FHI weight tuning (w1-w4)
├── src/
│   ├── model/                  # LLM inference layer
│   │   ├── llm_client.py       # Unified HuggingFace + Ollama client
│   │   ├── token_extractor.py  # Log-prob, attention, gradient extraction
│   │   └── prompt_templates.py # CoT, self-explanation, direct prompts
│   ├── xai/                    # Explainability methods
│   │   ├── attention_attribution.py  # Attention rollout (Abnar & Zuidema 2020)
│   │   ├── gradient_attribution.py   # Integrated Gradients via Captum
│   │   ├── shap_approximator.py      # Coalition sampling SHAP
│   │   └── attribution_utils.py      # Normalization & token cleaning
│   ├── perturbation/           # Causal perturbation engine
│   │   ├── token_masker.py     # Mask, delete, replace strategies
│   │   ├── output_comparator.py # JS divergence, semantic distance, ROUGE-L
│   │   └── causal_engine.py    # Orchestrates full causal impact analysis
│   ├── metrics/                # FHI component metrics
│   │   ├── aas.py              # Attribution Alignment Score
│   │   ├── cis.py              # Causal Impact Score
│   │   ├── ess.py              # Explanation Stability Score
│   │   ├── hcg.py              # Hallucination Confidence Gap
│   │   └── fhi.py              # Composite FHI + weight optimizer
│   ├── data/                   # Dataset loading & adversarial generation
│   │   ├── dataset_loader.py   # TriviaQA, HaluEval, MuSiQue loader
│   │   └── adversarial_generator.py  # False premise injection
│   ├── evaluation/             # Baselines & statistical evaluation
│   │   ├── baselines.py        # LogProb thresholding, Self-Consistency
│   │   └── evaluator.py        # Precision/Recall/F1/AUC-ROC comparison
│   └── visualization/          # Publication-quality plotting
│       └── plot_metrics.py     # KDE, scatter, ROC, radar charts
├── experiments/
│   └── run_pipeline.py         # Main entry point with MLflow tracking
├── dashboard/                  # Interactive web dashboard
│   ├── index.html
│   ├── style.css
│   └── app.js
├── paper/                      # LaTeX paper scaffold (IEEE format)
│   ├── main.tex
│   └── sections/
├── results/                    # Output data (gitignored)
├── EXPLANATION.md              # Detailed pipeline walkthrough
└── requirements.txt            # All Python dependencies
```

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Login to HuggingFace (for Gemma access)
hf auth login

# 4. Run test pipeline (5 samples)
python experiments/run_pipeline.py --test

# 5. Run on specific dataset
python experiments/run_pipeline.py --dataset halueval

# 6. Launch dashboard
python3 -m http.server 8080 -d dashboard/
# Open http://localhost:8080
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemma-2B-it (HuggingFace Transformers) |
| XAI | Captum (Integrated Gradients), Custom Attention Rollout |
| Semantic Similarity | SentenceTransformers (all-MiniLM-L6-v2) |
| Experiment Tracking | MLflow |
| Datasets | HuggingFace Datasets (TriviaQA, HaluEval, MuSiQue) |
| Visualization | Matplotlib, Seaborn, Chart.js (dashboard) |
| Compute | CPU-compatible (GPU/MPS optional) |

## Citation

```bibtex
@article{fhi2026,
  title={A Multi-Dimensional Evaluation of Explanation Faithfulness and Hallucination in Large Language Models},
  author={Your Name},
  year={2026}
}
```

## License

MIT
