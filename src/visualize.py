"""Publication-ready figures for the prompt-design study.

Every figure follows three rules: explicit axis labels with units, visible
uncertainty (95% confidence intervals), and a colorblind-safe palette with
distinct markers / line styles so the plots stay readable in grayscale.
Each figure is saved as vector PDF and as PNG at 300 DPI.

Example:
    python src/visualize.py --raw data/raw/test_run.jsonl
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import process_data  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
DPI = 300

PALETTE = sns.color_palette("colorblind")
STYLE = {
    "basic": {"color": PALETTE[0], "marker": "o", "linestyle": "-", "hatch": ""},
    "structured": {"color": PALETTE[1], "marker": "s", "linestyle": "--", "hatch": "//"},
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "phishing_detection_rate": "Phishing detection\nrate (recall)",
    "false_positive_rate": "False-positive\nrate",
}


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})


def save_figure(fig: plt.Figure, name: str, out_dir: Path = FIGURE_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension, kwargs in (("pdf", {}), ("png", {"dpi": DPI})):
        path = out_dir / f"{name}.{extension}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    return paths


def figure_latency_vs_length(by_length: pd.DataFrame, conditions: list[str]):
    """Fig. 1: median latency vs email length, with bootstrap 95% CI error bars."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for offset_index, condition in enumerate(conditions):
        data = by_length[by_length["condition"] == condition].sort_values("chars_median")
        style = STYLE[condition]
        # Small multiplicative shift keeps overlapping error bars distinguishable on a log axis.
        x = data["chars_median"].to_numpy() * (1.0 + 0.04 * offset_index)
        y = data["latency_median_ms"].to_numpy()
        yerr = np.vstack([y - data["ci_low_ms"].to_numpy(), data["ci_high_ms"].to_numpy() - y])
        ax.errorbar(
            x, y, yerr=yerr, color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
            linewidth=1.6, markersize=6, capsize=3, label=f"{condition} prompt",
        )
    ax.set_xscale("log")
    low, high = ax.get_xlim()
    ticks = [t for t in (50, 100, 200, 500, 1000, 2000, 5000) if low <= t <= high]
    if len(ticks) >= 2:
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Email length (characters, bin median, log scale)")
    ax.set_ylabel("Median latency per request (ms)")
    ax.legend(title="Condition (error bars: 95% bootstrap CI)", frameon=False)
    return fig


def figure_latency_boxplot(evaluation: pd.DataFrame, conditions: list[str]):
    """Fig. 2: latency distribution per condition; notches show the 95% CI of the median."""
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    data = [evaluation.loc[evaluation["condition"] == c, "latency_ms"].to_numpy() for c in conditions]
    box = ax.boxplot(
        data, notch=True, patch_artist=True, widths=0.5, showfliers=True,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, condition in zip(box["boxes"], conditions):
        patch.set_facecolor(STYLE[condition]["color"])
        patch.set_hatch(STYLE[condition]["hatch"])
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels([f"{c}\n(n = {len(d)})" for c, d in zip(conditions, data)])
    ax.set_ylabel("Latency per request (ms)")
    ax.set_xlabel("Prompt condition (notch: 95% CI of the median)")
    positive = np.concatenate(data)
    if positive.min() > 0 and positive.max() / positive.min() > 20:
        ax.set_yscale("log")
        ax.set_ylabel("Latency per request (ms, log scale)")
    return fig


def figure_metrics_ci(metrics: pd.DataFrame, conditions: list[str]):
    """Fig. 3: accuracy, recall and FPR per condition with Wilson 95% CI."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    metric_names = list(METRIC_LABELS)
    positions = np.arange(len(metric_names))
    for offset_index, condition in enumerate(conditions):
        style = STYLE[condition]
        subset = metrics[metrics["condition"] == condition].set_index("metric").loc[metric_names]
        shift = (offset_index - (len(conditions) - 1) / 2) * 0.16
        y = subset["value"].to_numpy()
        yerr = np.vstack([y - subset["ci_low"].to_numpy(), subset["ci_high"].to_numpy() - y])
        ax.errorbar(
            positions + shift, y, yerr=yerr, color=style["color"], marker=style["marker"],
            linestyle="none", markersize=8, capsize=4, linewidth=1.6, label=f"{condition} prompt",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metric_names])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Proportion of emails (0-1)")
    ax.set_xlabel("Metric (error bars: Wilson 95% CI)")
    ax.legend(frameon=False, loc="best")
    return fig


def make_all(result: dict, out_dir: Path = FIGURE_DIR) -> list[Path]:
    apply_style()
    conditions = result["conditions"]
    saved = []
    for name, fig in (
        ("fig1_latency_vs_length", figure_latency_vs_length(result["by_length"], conditions)),
        ("fig2_distribution_boxplot", figure_latency_boxplot(result["evaluation"], conditions)),
        ("fig3_metrics_with_ci", figure_metrics_ci(result["metrics"], conditions)),
    ):
        saved.extend(save_figure(fig, name, out_dir))
        plt.close(fig)
    return saved


def main() -> int:
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/raw/test_run.jsonl", help="Raw .jsonl log from benchmark.py")
    parser.add_argument("--out", default=str(FIGURE_DIR), help="Directory for figures")
    args = parser.parse_args()
    try:
        raw_path = process_data.resolve_path(args.raw)
        result = process_data.process(raw_path, PROJECT_ROOT / "data" / "processed")
        saved = make_all(result, Path(args.out))
    except (OSError, ValueError) as error:
        print(f"Visualization failed: {error}", file=sys.stderr)
        return 1
    for path in saved:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
