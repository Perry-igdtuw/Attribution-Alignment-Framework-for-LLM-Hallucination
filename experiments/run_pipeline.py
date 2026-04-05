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
from typing import List, Optional, Dict, Any

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
from src.perturbation.causal_engine import CausalEngine
from src.data.dataset_loader import load_dataset
from src.data.adversarial_generator import AdversarialGenerator
from src.evaluation.baselines import BaselineEvaluator
from src.evaluation.evaluator import SystemEvaluator

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


def run_single_sample(
    client: LLMClient,
    sample: dict,
    attention_attr: AttentionAttribution,
    gradient_attr: GradientAttribution,
    causal_engine: CausalEngine,
    baseline_evaluator: BaselineEvaluator,
    weight_cfg: dict,
    exp_cfg: dict,
    compute_gradients: bool = False,
    skip_sc_baseline: bool = False,
) -> Dict[str, Any]:
    """
    Run the full FHI pipeline on a single sample, plus baselines.
    Returns a dictionary of all results.
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
        output_shifts = causal_engine.measure_causal_impact(
            original_question=question,
            original_response_text=response.answer,
            original_explanation=response.explanation,
            original_log_probs=response.token_log_probs or None,
            attribution_result=primary_attr,
            top_k=top_k,
        )

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

    # ── Step 8: Baselines ─────────────────────────────────────────────────────
    # Evaluate Log-Prob baseline
    logprob_hl_pred = baseline_evaluator.evaluate_logprob(
        response.token_log_probs or [], threshold=-0.5
    )
    
    # Evaluate Self-Consistency baseline (skip in fast/test mode — very expensive on CPU)
    sc_risk = 0.0
    if not skip_sc_baseline:
        sc_risk = baseline_evaluator.evaluate_self_consistency(
            question, n_samples=3
        )

    return {
        "fhi_result": fhi_result,
        "logprob_hallucination": logprob_hl_pred,
        "self_consistency_risk": sc_risk,
        "is_adversarial": sample.get("is_adversarial", False)
    }


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

    # Setup Phase 4 and Phase 6 Orchestrators
    masker = TokenMasker()
    comparator = OutputComparator()
    causal_engine = CausalEngine(
        llm_client=client, 
        masker=masker, 
        comparator=comparator,
        strategies=exp_cfg["experiment"]["perturbation_strategies"]
    )
    baseline_evaluator = BaselineEvaluator(llm_client=client)

    # ── MLflow tracking ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(exp_cfg["experiment"]["mlflow_tracking_uri"])
    mlflow.set_experiment(exp_cfg["experiment"]["name"])

    dataset_names = (
        [args.dataset]
        if args.dataset
        else ["halueval"]  # Prioritize HaluEval as it has ground truth
    )

    all_results: List[Dict[str, Any]] = []

    with mlflow.start_run(run_name=f"fhi_evaluation_{model_cfg['model']['backend']}"):
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
            seed = exp_cfg["experiment"].get("seed", 42)
            samples = load_dataset(dataset_name, n_samples, seed)
            
            # (Optional) We can make the first sample adversarial if testing
            if args.test and samples:
                samples[0] = AdversarialGenerator.inject_false_premise(samples[0])

            for sample in track(samples, description=f"Processing {dataset_name}..."):
                try:
                    result = run_single_sample(
                        client=client,
                        sample=sample,
                        attention_attr=attention_attr,
                        gradient_attr=gradient_attr,
                        causal_engine=causal_engine,
                        baseline_evaluator=baseline_evaluator,
                        weight_cfg=weight_cfg,
                        exp_cfg=exp_cfg,
                        compute_gradients=False,
                        skip_sc_baseline=args.test,  # Skip SC in test mode (too slow on CPU)
                    )
                    all_results.append(result)
                except Exception as e:
                    import traceback
                    logger.error(f"Error on sample {sample.get('id', '?')}: {e}\n{traceback.format_exc()}")
                    continue

        # ── Log aggregate metrics to MLflow ──────────────────────────────────
        if all_results:
            fhi_res = [r["fhi_result"] for r in all_results]
            fhi_scores = [r.fhi for r in fhi_res]
            
            mlflow.log_metrics({
                "mean_fhi": float(np.mean(fhi_scores)),
                "mean_aas": float(np.mean([r.aas for r in fhi_res])),
                "mean_cis": float(np.mean([r.cis for r in fhi_res])),
                "mean_ess": float(np.mean([r.ess for r in fhi_res])),
                "mean_hcg": float(np.mean([r.hcg for r in fhi_res])),
            })

            # ── Evaluate hallucination detection (Phase 6 Evaluator) ────────
            labeled = [r for r in all_results if r["fhi_result"].true_hallucination is not None]
            if labeled:
                y_true = [r["fhi_result"].true_hallucination for r in labeled]
                fhi_scores = [r["fhi_result"].fhi for r in labeled]
                logprob_preds = [r["logprob_hallucination"] for r in labeled]
                sc_scores = [r["self_consistency_risk"] for r in labeled]
                
                eval_report = SystemEvaluator.evaluate(
                    y_true=y_true,
                    fhi_scores=fhi_scores,
                    logprob_preds=logprob_preds,
                    self_consistency_scores=sc_scores,
                    fhi_threshold=exp_cfg["experiment"]["fhi_threshold"]
                )
                
                # Log metrics cleanly
                mlflow.log_metrics({
                    "FHI_F1": eval_report["FHI"]["macro avg"]["f1-score"],
                    "LogProb_F1": eval_report["LogProb"]["macro avg"]["f1-score"],
                })
                
                if eval_report.get("FHI_AUC") is not None:
                    console.print(f"\n[bold]AUC-ROC[/bold] | FHI: {eval_report['FHI_AUC']:.2f} vs Self-Consistency: {eval_report['SelfConsistency_AUC']:.2f}")

                console.print("\n[bold]FHI Classification Report:[/bold]")
                console.print(eval_report["FHI"])

    # ── Save results to JSON ──────────────────────────────────────────────────
    out_dir = Path(exp_cfg["experiment"]["output_dir"]) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"fhi_results_{exp_cfg['experiment']['name']}.json"

    results_json = []
    for r in all_results:
        fhi_r = r["fhi_result"]
        results_json.append({
            "sample_id": fhi_r.sample_id,
            "fhi": fhi_r.fhi, "aas": fhi_r.aas, "cis": fhi_r.cis,
            "ess": fhi_r.ess, "hcg": fhi_r.hcg,
            "predicted_hallucination": fhi_r.predicted_hallucination,
            "true_hallucination": fhi_r.true_hallucination,
            "baseline_logprob": r["logprob_hallucination"],
            "baseline_sc_risk": r["self_consistency_risk"],
            "adversarial_injected": r["is_adversarial"]
        })
        
    with open(out_file, "w") as f:
        json.dump(results_json, f, indent=2)

    console.print(f"\n[green]Results saved to:[/green] {out_file}")

    # ── Summary table ─────────────────────────────────────────────────────────
    table = Table(title="FHI Pipeline Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="yellow")
    
    fhi_res = [r["fhi_result"] for r in all_results]
    for metric, vals in [
        ("FHI", [r.fhi for r in fhi_res]),
        ("AAS", [r.aas for r in fhi_res]),
        ("CIS", [r.cis for r in fhi_res]),
        ("ESS", [r.ess for r in fhi_res]),
        ("HCG", [r.hcg for r in fhi_res]),
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
