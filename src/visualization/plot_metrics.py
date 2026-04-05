"""
plot_metrics.py
---------------
Generates publication-quality visualizations for the FHI pipeline.
Consumes the JSON output from run_pipeline.py and creates statistical plots.
"""

import json
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Aesthetic styling for publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_results(json_path: Path) -> pd.DataFrame:
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        logger.error(f"File not found: {json_path}")
        return pd.DataFrame()


def plot_fhi_distributions(df: pd.DataFrame, output_dir: Path):
    """
    Plots the Kernel Density Estimation (KDE) of the FHI scores,
    split by True Hallucination vs Factual (if labels exist).
    """
    plt.figure(figsize=(8, 5))
    
    # If we have labels, split the distribution
    if "true_hallucination" in df.columns and df["true_hallucination"].notna().any():
        labeled_df = df.dropna(subset=["true_hallucination"])[["fhi", "true_hallucination"]]
        factual_df = labeled_df[labeled_df["true_hallucination"] == False]
        hallu_df = labeled_df[labeled_df["true_hallucination"] == True]
        
        sns.kdeplot(data=factual_df, x="fhi", fill=True, label="Factual", color="forestgreen", alpha=0.5)
        sns.kdeplot(data=hallu_df, x="fhi", fill=True, label="Hallucination", color="crimson", alpha=0.5)
        plt.legend(title="Ground Truth")
    else:
        sns.kdeplot(data=df, x="fhi", fill=True, color="royalblue", alpha=0.5)

    plt.title("Distribution of Faithfulness-Hallucination Index (FHI)")
    plt.xlabel("FHI Score (0 = Hallucination, 1 = Faithful)")
    plt.ylabel("Density")
    plt.xlim(0, 1)    
    plt.savefig(output_dir / "fhi_distribution.png")
    plt.close()


def plot_metric_correlation(df: pd.DataFrame, output_dir: Path):
    """
    Creates a scatter plot showing the correlation between Attribution
    Alignment (AAS) and Causal Impact (CIS).
    """
    plt.figure(figsize=(7, 6))
    
    # Drop NaNs just in case
    clean_df = df.dropna(subset=["aas", "cis"])
    if clean_df.empty:
        logger.warning("No data available for AAS vs CIS scatter.")
        return

    sns.scatterplot(
        data=clean_df, 
        x="aas", 
        y="cis", 
        hue="fhi", 
        palette="viridis",
        size="ess",
        sizes=(20, 200),
        alpha=0.7
    )
    
    # Calculate pearson correlation
    corr = np.corrcoef(clean_df["aas"], clean_df["cis"])[0, 1]
    
    plt.title(f"AAS vs CIS Correlation (r={corr:.2f})")
    plt.xlabel("Attribution Alignment Score (AAS)")
    plt.ylabel("Causal Impact Score (CIS)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Legend formatting
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(output_dir / "aas_vs_cis_scatter.png")
    plt.close()


def plot_roc_curves(df: pd.DataFrame, output_dir: Path):
    """
    Plots the ROC curves comparing FHI against baselines 
    (LogProb, Self-Consistency) for Hallucination detection.
    """
    # Requires labeled ground truth
    if "true_hallucination" not in df.columns or df["true_hallucination"].isna().all():
        logger.warning("No ground truth labels found. Skipping ROC plot.")
        return
        
    labeled_df = df.dropna(subset=["true_hallucination"])
    y_true = np.array(labeled_df["true_hallucination"], dtype=bool)

    # If only one class is present in the batch, ROC cannot be drawn.
    if len(set(y_true)) < 2:
        logger.warning("Only one class found in ground truth. Skipping ROC plot.")
        return

    plt.figure(figsize=(7, 6))
    
    # FHI ROC (Lower FHI = Higher probability of hallucination)
    y_scores_fhi = 1.0 - labeled_df["fhi"]
    fpr_fhi, tpr_fhi, _ = roc_curve(y_true, y_scores_fhi)
    roc_auc_fhi = auc(fpr_fhi, tpr_fhi)
    plt.plot(fpr_fhi, tpr_fhi, color='darkorange', lw=2, label=f'FHI (AUC = {roc_auc_fhi:.2f})')

    # Baseline: LogProb Risk (Since we only stored True/False, this will be somewhat rigid)
    if "baseline_logprob" in labeled_df.columns:
        y_scores_lp = np.array(labeled_df["baseline_logprob"], dtype=float)
        fpr_lp, tpr_lp, _ = roc_curve(y_true, y_scores_lp)
        roc_auc_lp = auc(fpr_lp, tpr_lp)
        plt.plot(fpr_lp, tpr_lp, color='navy', lw=2, linestyle='--', label=f'LogProb (AUC = {roc_auc_lp:.2f})')

    # Baseline: Self Consistency Risk
    if "baseline_sc_risk" in labeled_df.columns:
        y_scores_sc = np.array(labeled_df["baseline_sc_risk"], dtype=float)
        fpr_sc, tpr_sc, _ = roc_curve(y_true, y_scores_sc)
        roc_auc_sc = auc(fpr_sc, tpr_sc)
        plt.plot(fpr_sc, tpr_sc, color='forestgreen', lw=2, linestyle='-.', label=f'Self-Consistency (AUC = {roc_auc_sc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Hallucination Detection')
    plt.legend(loc="lower right")
    plt.savefig(output_dir / "roc_curve_comparison.png")
    plt.close()


def generate_radar_chart(df: pd.DataFrame, output_dir: Path):
    """
    Generates a Radar (Spider) Chart showing the mean scores for all metrics.
    Useful for seeing the 'profile' of the model.
    """
    labels = np.array(['AAS', 'CIS', 'ESS', 'HCG'])
    
    # Drop NaNs
    metrics_df = df[["aas", "cis", "ess", "hcg"]].dropna()
    if metrics_df.empty:
        return
        
    stats = metrics_df.mean().values
    
    # We want everything on a 0-1 scale. HCG could be technically -1 to 1,
    # but practically we clip it in the FHI pipeline. We will just use the raw mean.
    
    # Close the loop
    stats_closed = np.concatenate((stats, [stats[0]]))
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles_closed = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles_closed, stats_closed, color='royalblue', alpha=0.25)
    ax.plot(angles_closed, stats_closed, color='royalblue', linewidth=2)
    
    # Set the labels
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    
    plt.title("Model Trust Profile (Mean Metrics)", size=16, y=1.1)
    plt.savefig(output_dir / "metrics_radar_chart.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Visualizations for FHI")
    parser.add_argument("--input", type=str, required=True, help="Path to raw JSON results file")
    parser.add_argument("--output_dir", type=str, default="results/plots", help="Directory to save plots")
    args = parser.parse_args()

    json_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading results from {json_path}")
    df = load_results(json_path)

    if df.empty:
        logger.error("No data found. Exiting.")
        return

    logger.info("Generating FHI Distribution...")
    plot_fhi_distributions(df, output_dir)
    
    logger.info("Generating AAS vs CIS Correlation Scatter...")
    plot_metric_correlation(df, output_dir)
    
    logger.info("Generating ROC Curves comparing Baselines...")
    plot_roc_curves(df, output_dir)
    
    logger.info("Generating Model Radar Chart...")
    generate_radar_chart(df, output_dir)

    logger.info(f"All plots successfully saved to: {output_dir}")


if __name__ == "__main__":
    main()
