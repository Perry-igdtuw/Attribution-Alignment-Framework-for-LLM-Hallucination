# Pipeline Explanation: FHI Evaluation

This document explains exactly what the Faithfulness-Hallucination Index (FHI) pipeline is doing behind the scenes when you run `python experiments/run_pipeline.py`.

## The Core Concept

We hypothesize that hallucinations in Large Language Models (LLMs) happen when the model's generated explanation is causally disconnected from its actual internal reasoning. If an LLM is truly reasoning, removing key words from its explanation should change its final answer. If removing the explanation doesn't change anything, the explanation was just a "post-hoc rationalization" (a hallucination feature).

The pipeline measures this across 5 samples (for testing) using multiple advanced metrics.

## Step-by-Step Breakdown (Per Question)

When the pipeline processes a single question (e.g., from TriviaQA), it executes the following steps:

### 1. Generation (`llm_client.py`)
- The model (Gemma-2B) is given the prompt: *"Answer this question... now explain why."*
- It generates the answer and the explanation.
- We also extract its internal **log-probabilities** (token confidences) and **attention weights** (how heavily it looked at different words).

### 2. Attribution Analysis (`attention_attribution.py` / `gradient_attribution.py`)
- The system asks: *Which words in the input prompt were actually important to the model?*
- It computes this using **Attention Rollout**, tracking attention flow through the transformer layers.
- It identifies the top-K most heavily attended words.

### 3. Metric 1: AAS (Attribution Alignment Score)
- **What it is:** Measures overlap between the explanation words and the highly-attended words.
- **Why:** If the model says "I chose Paris because it's the capital of France," but it only "paid attention" to the word "Eiffel Tower" internally, the explanation isn't aligned with its reasoning.
- **Formula:** Jaccard overlap between the explanation words and the top-K attribution tokens.

### 4. Metric 2: CIS (Causal Impact Score)
- **What it is:** The heart of the pipeline. It tests causality by perturbing the prompt.
- **How it works (`token_masker.py` and `output_comparator.py`):**
  1. It takes the top explanation words and removes/masks them in the prompt.
  2. It forces the model to answer again.
  3. It measures how much the answer changed using:
     - **JS Divergence:** Did the probability distribution change?
     - **Semantic Distance:** Did the meaning of the answer change (SentenceTransformers)?
     - **ROUGE-L Delta:** Did the exact wording change?
- **Why:** High Causal Impact means the explanation actually mattered. Low impact means the model produced the answer regardless of the explanation (high hallucination risk).

### 5. Metric 3: ESS (Explanation Stability Score)
- **What it is:** Checks if the model is guessing by asking it the same thing multiple times.
- **How it works:** The model is asked the question 5 times. We measure the semantic similarity of the 5 explanations.
- **Why:** A hallucinating model often gives inconsistent reasons, while grounded knowledge produces stable explanations.

### 6. Metric 4: HCG (Hallucination Confidence Gap)
- **What it is:** Penalizes "confidently wrong" answers.
- **Formula:** `Confidence - Correctness_Score` (where correctness is Token-F1 against the ground truth).
- **Why:** Hallucinations are characterized by the model being highly confident despite being factually incorrect.

### 7. Final Score: FHI (Faithfulness-Hallucination Index)
- **Formula:** `w1*AAS + w2*CIS + w3*ESS - w4*HCG`
- All these signals are combined. A high FHI score (> 0.5) means the model was faithful and grounded. A low score means hallucination was detected.

## What is happening right now during execution?
Because we are running locally on a CPU, and each sample involves:
- 1 initial generation
- 5 stability generation runs
- 3 causal perturbation generation runs (mask, delete, replace)
- Semantic embedding extraction for evaluation

It performs ~9 LLM generation passes per single question. This is computationally intense but provides an extremely rigorous, publication-level evaluation of model faithfulness.
