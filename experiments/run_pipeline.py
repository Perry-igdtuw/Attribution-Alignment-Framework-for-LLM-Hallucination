"""
run_pipeline.py
---------------
Main entry point for the FHI evaluation pipeline.

Usage:
    # Full run on all datasets:
    python experiments/run_pipeline.py

    # Quick test on 5 samples (for debugging):
    python experiments/run_pipeline.py --test

    # Run specific dataset only:
    python experiments/run_pipeline.py --dataset trivia_qa

    # Use Ollama backend instead of HuggingFace:
    python experiments/run_pipeline.py --backend ollama
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import mlflow
import numpy as np
import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.llm_client import LLMClient
from src.xai.attention_attribution import AttentionAttribution
from src.xai.gradient_attribution import GradientAttribution
from src.perturbation.token_masker import TokenMasker, MaskStrategy
from src.perturbation.output_comparator import OutputComparator
from src.metrics.aas import compute_aas
from src.metrics.cis import compute_cis
from src.metrics.ess import compute_ess_from_responses
from src.metrics.hcg import compute_hcg
from src.metrics.fhi import compute_fhi, FHIResult

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config():
    base = Path(__file__).parent.parent / "config"
    with open(base / "model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open(base / "experiment_config.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    with open(base / "metric_weights.yaml") as f:
        weight_cfg = yaml.safe_load(f)
    return model_cfg, exp_cfg, weight_cfg


def load_dataset(dataset_name: str, n_samples: int, exp_cfg: dict):
    """Load a HuggingFace dataset and return list of {question, answer} dicts."""
    from datasets import load_dataset as hf_load

    console.print(f"[cyan]Loading dataset: {dataset_name}[/cyan]")

    if dataset_name == "trivia_qa":
        ds = hf_load("trivia_qa", "rc.nocontext", split="validation")
        samples = []
        for item in ds.select(range(n_samples)):
            samples.append({
                "id": item["question_id"],
                "question": item["question"],
                "gold_answer": item["answer"]["value"],
                "is_hallucination": False,  # factual dataset = no hallucination label
            })
        return samples

    elif dataset_name == "halueval":
        ds = hf_load("pminervini/HaluEval", "qa_samples", split="data")
        samples = []
        for i, item in enumerate(ds.select(range(n_samples))):
            # HaluEval schema: question, answer, hallucination ('yes'/'no'), knowledge
            # 'answer' is the model-generated answer; 'hallucination' tells if it's hallucinated
            samples.append({
                "id": f"halueval_{i}",
                "question": item["question"],
                "gold_answer": item["knowledge"],      # use knowledge as reference context
                "is_hallucination": item["hallucination"] == "yes",
            })
            if len(samples) >= n_samples:
                break
        return samples

    else:
        # Fallback: minimal synthetic test data
        logger.warning(f"Dataset {dataset_name} not found. Using synthetic test data.")
        return [
            {"id": f"test_{i}", "question": f"Test question {i}?",
             "gold_answer": "test answer", "is_hallucination": False}
            for i in range(min(n_samples, 5))
        ]


def run_single_sample(
    client: LLMClient,
    sample: dict,
    attention_attr: AttentionAttribution,
    gradient_attr: GradientAttribution,
    masker: TokenMasker,
    comparator: OutputComparator,
    weight_cfg: dict,
    exp_cfg: dict,
    compute_gradients: bool = False,
) -> FHIResult:
    """
    Run the full FHI pipeline on a single sample.
    Returns a FHIResult object.
    """
    question = sample["question"]
    gold_answer = sample.get("gold_answer", "")
    true_hallucination = sample.get("is_hallucination", None)

    # ── Step 1: Generate answer + explanation ─────────────────────────────────
    response = client.generate(
        question=question,
        strategy="self_explain",
        compute_gradients=compute_gradients,
    )

    # ── Step 2: XAI Attribution ───────────────────────────────────────────────
    attribution_results = []

    # Attention attribution (requires HF backend)
    attn_result = attention_attr.compute(response)
    if attn_result:
        attribution_results.append(attn_result)

    # Gradient attribution (optional, slower)
    if compute_gradients and response.backend == "huggingface":
        grad_result = gradient_attr.compute_grad_x_input(response)
        if grad_result:
            attribution_results.append(grad_result)

    # ── Step 3: Compute AAS ───────────────────────────────────────────────────
    top_k = exp_cfg["experiment"]["attribution_top_k"]
    aas = compute_aas(response.explanation, attribution_results, top_k=top_k)

    # ── Step 4: Causal Perturbation → CIS ─────────────────────────────────────
    output_shifts = []
    if attribution_results:
        primary_attr = attribution_results[0]  # Use attention as primary

        for strategy_str in exp_cfg["experiment"]["perturbation_strategies"]:
            strategy = MaskStrategy(strategy_str)
            masked = masker.mask_explanation_in_prompt(
                prompt=question,
                explanation=response.explanation,
                attribution_result=primary_attr,
                strategy=strategy,
            )

            # Generate output on perturbed prompt
            perturbed_resp = client.generate(masked.masked_text, strategy="direct")

            shift = comparator.compare(
                original_output=response.answer,
                perturbed_output=perturbed_resp.answer,
                strategy=strategy_str,
                original_log_probs=response.token_log_probs or None,
                perturbed_log_probs=perturbed_resp.token_log_probs or None,
            )
            output_shifts.append(shift)

    cis = compute_cis(output_shifts)

    # ── Step 5: Explanation Stability → ESS ──────────────────────────────────
    n_runs = client.config["model"]["n_stability_runs"]
    stability_responses = client.generate_multiple(
        question, n=n_runs, strategy="self_explain"
    )
    ess = compute_ess_from_responses(stability_responses)

    # ── Step 6: Hallucination Confidence Gap → HCG ───────────────────────────
    hcg = compute_hcg(
        model_confidence=response.answer_confidence,
        predicted_answer=response.answer,
        gold_answer=gold_answer,
        correctness_method="token_f1",
    )

    # ── Step 7: Compute FHI ───────────────────────────────────────────────────
    w = weight_cfg["weights"]
    fhi_result = compute_fhi(
        aas=aas, cis=cis, ess=ess, hcg=hcg,
        w1=w["w1_aas"], w2=w["w2_cis"], w3=w["w3_ess"], w4=w["w4_hcg"],
        threshold=exp_cfg["experiment"]["fhi_threshold"],
        sample_id=sample.get("id", "unknown"),
        true_hallucination=true_hallucination,
    )

    return fhi_result


def main(args):
    model_cfg, exp_cfg, weight_cfg = load_config()

    # Override backend from CLI
    if args.backend:
        model_cfg["model"]["backend"] = args.backend

    n_samples = 5 if args.test else exp_cfg["experiment"]["n_samples"]

    console.print(
        f"[bold green]FHI Pipeline[/bold green] | "
        f"Backend: {model_cfg['model']['backend']} | "
        f"Samples: {n_samples}"
    )

    # ── Initialize clients ────────────────────────────────────────────────────
    console.print("[yellow]Loading LLM...[/yellow]")
    client = LLMClient(model_cfg)

    attention_attr = AttentionAttribution(top_k=exp_cfg["experiment"]["attribution_top_k"])
    gradient_attr = None
    if model_cfg["model"]["backend"] == "huggingface":
        gradient_attr = GradientAttribution(
            model=client._model,
            tokenizer=client._tokenizer,
            device=model_cfg["model"]["device"],
        )

    masker = TokenMasker()
    comparator = OutputComparator()

    # ── MLflow tracking ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(exp_cfg["experiment"]["mlflow_tracking_uri"])
    mlflow.set_experiment(exp_cfg["experiment"]["name"])

    dataset_names = (
        [args.dataset]
        if args.dataset
        else ["trivia_qa", "halueval"]
    )

    all_results: List[FHIResult] = []

    with mlflow.start_run(run_name=f"fhi_{model_cfg['model']['backend']}"):
        mlflow.log_params({
            "backend": model_cfg["model"]["backend"],
            "model": model_cfg["model"]["hf_model_id"],
            "n_samples": n_samples,
            **{f"w{i+1}": v for i, v in enumerate([
                weight_cfg["weights"]["w1_aas"],
                weight_cfg["weights"]["w2_cis"],
                weight_cfg["weights"]["w3_ess"],
                weight_cfg["weights"]["w4_hcg"],
            ])},
        })

        for dataset_name in dataset_names:
            samples = load_dataset(dataset_name, n_samples, exp_cfg)

            for sample in track(samples, description=f"Processing {dataset_name}..."):
                try:
                    result = run_single_sample(
                        client=client,
                        sample=sample,
                        attention_attr=attention_attr,
                        gradient_attr=gradient_attr,
                        masker=masker,
                        comparator=comparator,
                        weight_cfg=weight_cfg,
                        exp_cfg=exp_cfg,
                        compute_gradients=False,  # Enable for full XAI run
                    )
                    all_results.append(result)
                except Exception as e:
                    import traceback
                    logger.error(f"Error on sample {sample.get('id', '?')}: {e}\n{traceback.format_exc()}")
                    continue

        # ── Log aggregate metrics to MLflow ──────────────────────────────────
        if all_results:
            fhi_scores = [r.fhi for r in all_results]
            mlflow.log_metrics({
                "mean_fhi": float(np.mean(fhi_scores)),
                "mean_aas": float(np.mean([r.aas for r in all_results])),
                "mean_cis": float(np.mean([r.cis for r in all_results])),
                "mean_ess": float(np.mean([r.ess for r in all_results])),
                "mean_hcg": float(np.mean([r.hcg for r in all_results])),
            })

            # ── Evaluate hallucination detection ──────────────────────────────
            labeled = [r for r in all_results if r.true_hallucination is not None]
            if labeled:
                from sklearn.metrics import classification_report
                y_true = [r.true_hallucination for r in labeled]
                y_pred = [r.predicted_hallucination for r in labeled]
                report = classification_report(y_true, y_pred, output_dict=True)
                mlflow.log_metrics({
                    "precision": report["weighted avg"]["precision"],
                    "recall": report["weighted avg"]["recall"],
                    "f1": report["weighted avg"]["f1-score"],
                })
                console.print("\n[bold]Classification Report:[/bold]")
                console.print(classification_report(y_true, y_pred))

    # ── Save results to JSON ──────────────────────────────────────────────────
    out_dir = Path(exp_cfg["experiment"]["output_dir"]) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"fhi_results_{exp_cfg['experiment']['name']}.json"

    results_json = [
        {
            "sample_id": r.sample_id,
            "fhi": r.fhi, "aas": r.aas, "cis": r.cis,
            "ess": r.ess, "hcg": r.hcg,
            "predicted_hallucination": r.predicted_hallucination,
            "true_hallucination": r.true_hallucination,
        }
        for r in all_results
    ]
    with open(out_file, "w") as f:
        json.dump(results_json, f, indent=2)

    console.print(f"\n[green]Results saved to:[/green] {out_file}")

    # ── Summary table ─────────────────────────────────────────────────────────
    table = Table(title="FHI Pipeline Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="yellow")
    for metric, vals in [
        ("FHI", [r.fhi for r in all_results]),
        ("AAS", [r.aas for r in all_results]),
        ("CIS", [r.cis for r in all_results]),
        ("ESS", [r.ess for r in all_results]),
        ("HCG", [r.hcg for r in all_results]),
    ]:
        table.add_row(metric, f"{np.mean(vals):.4f}", f"{np.std(vals):.4f}")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FHI Evaluation Pipeline")
    parser.add_argument("--test", action="store_true", help="Run on 5 samples only")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument("--backend", type=str, default=None, choices=["huggingface", "ollama"])
    args = parser.parse_args()
    main(args)
