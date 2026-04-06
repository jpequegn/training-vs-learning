"""Generate visualizations for the Training vs. Learning analysis."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")


def load_results():
    with open(RESULTS_DIR / "probe_factual_recall.json") as f:
        factual = json.load(f)
    with open(RESULTS_DIR / "probe_reasoning.json") as f:
        reasoning = json.load(f)
    with open(RESULTS_DIR / "probe_ood.json") as f:
        ood = json.load(f)
    return factual, reasoning, ood


def plot_probe_summary(factual, reasoning, ood):
    """Bar chart comparing all models across all probe dimensions."""
    models = ["gpt2-base", "gpt2-lora-completion", "gpt2-lora-qa"]
    labels = ["Base GPT-2", "FT Completion", "FT Q&A"]
    colors = ["#4A90D9", "#E74C3C", "#F39C12"]

    metrics = {
        "Factual Recall\n(accuracy %)": [factual[m]["accuracy"] * 100 for m in models],
        "Hallucination\nRate (%)": [factual[m]["hallucination_rate"] * 100 for m in models],
        "Reasoning\nScore (/5)": [reasoning[m]["avg_score"] for m in models],
        "OOD Confidence": [ood[m]["overall"]["avg_confidence"] for m in models],
        "Domain\nContamination (%)": [ood[m]["off_topic"]["contamination_rate"] * 100 for m in models],
    }

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle("Training vs. Learning: Probe Results Summary", fontsize=16, fontweight="bold", y=1.02)

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_title(metric_name, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out = RESULTS_DIR / "probe_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_perplexity():
    """Perplexity reduction chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    formats = ["Completion", "Q&A"]
    base_ppl = [50.51, 64.00]
    ft_ppl = [12.25, 5.36]

    x = np.arange(len(formats))
    width = 0.3

    bars1 = ax.bar(x - width/2, base_ppl, width, label="Base GPT-2", color="#4A90D9", edgecolor="white")
    bars2 = ax.bar(x + width/2, ft_ppl, width, label="Fine-tuned", color="#E74C3C", edgecolor="white")

    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title("Perplexity Reduction: What Fine-tuning Actually Improves", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(formats)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add reduction annotations
    for i, (b, f) in enumerate(zip(base_ppl, ft_ppl)):
        pct = (1 - f / b) * 100
        ax.annotate(f"-{pct:.0f}%", xy=(i + width/2, f), xytext=(i + 0.5, f + 10),
                    fontsize=11, fontweight="bold", color="#27AE60",
                    arrowprops=dict(arrowstyle="->", color="#27AE60"))

    plt.tight_layout()
    out = RESULTS_DIR / "perplexity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_training_vs_learning():
    """The money chart: what training improved vs what learning requires."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = [
        "Perplexity\n(style match)",
        "Factual\nRecall",
        "Reasoning",
        "Calibration",
        "Domain\nContainment",
    ]

    # Normalized scores: positive = improvement, negative = degradation
    # relative to base model
    ft_completion = [
        75.7,   # perplexity reduction
        -66,    # accuracy went from 5% to 1.7% (worse)
        9,      # 2.87 -> 3.13 (marginal)
        -40,    # confidence increased on unknowns (worse calibration)
        0,      # no contamination
    ]

    ft_qa = [
        91.6,   # perplexity reduction
        0,      # same accuracy
        2,      # 2.87 -> 2.93 (negligible)
        5,      # slightly less confident (neutral)
        -30,    # 30% contamination (worse)
    ]

    x = np.arange(len(categories))
    width = 0.3

    bars1 = ax.barh(x - width/2, ft_completion, width, label="FT Completion",
                     color=["#27AE60" if v > 0 else "#E74C3C" for v in ft_completion], alpha=0.8)
    bars2 = ax.barh(x + width/2, ft_qa, width, label="FT Q&A",
                     color=["#27AE60" if v > 0 else "#E74C3C" for v in ft_qa], alpha=0.5)

    ax.set_yticks(x)
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel("Change from Base Model (%)", fontsize=10)
    ax.set_title("Training vs. Learning\nGreen = improved, Red = degraded", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotations
    ax.annotate("TRAINING\n(pattern matching)", xy=(60, 4.3), fontsize=11,
                fontweight="bold", color="#27AE60", ha="center")
    ax.annotate("LEARNING\n(not achieved)", xy=(-40, -0.7), fontsize=11,
                fontweight="bold", color="#E74C3C", ha="center")

    plt.tight_layout()
    out = RESULTS_DIR / "training_vs_learning.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    factual, reasoning, ood = load_results()
    plot_probe_summary(factual, reasoning, ood)
    plot_perplexity()
    plot_training_vs_learning()
    print("\nAll visualizations generated.")


if __name__ == "__main__":
    main()
