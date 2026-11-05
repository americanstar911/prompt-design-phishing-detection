"""Clean raw benchmark logs and compute the statistics reported in the study.

Reads the raw .jsonl file written by src/benchmark.py, discards warm-up requests,
and writes aggregated tables to data/processed/.

Example:
    python src/process_data.py --raw data/raw/test_run.jsonl --out data/processed --update-readme
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
Z_95 = 1.959963984540054
BASELINE = "basic"
PROPOSED = "structured"
README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"

np.random.seed(SEED)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ---------------------------------------------------------------- ingestion

def load_raw(path: Path) -> pd.DataFrame:
    """Load a .jsonl benchmark log into a DataFrame."""
    records = []
    with path.open("r", encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}: line {number} is not valid JSON ({error})") from error
    if not records:
        raise ValueError(f"{path} contains no records.")
    frame = pd.DataFrame(records)
    required = {"phase", "condition", "id", "true_label", "predicted_label", "latency_ms", "email_chars"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    return frame


def split_warmup(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return (evaluation records, number of discarded warm-up records)."""
    warmup = frame[frame["phase"] == "warmup"]
    evaluation = frame[frame["phase"] == "eval"].copy()
    return evaluation.reset_index(drop=True), int(len(warmup))


def keep_complete_ids(evaluation: pd.DataFrame, conditions: list[str]) -> tuple[pd.DataFrame, int]:
    """Keep only emails that have a result in every condition (needed for paired tests)."""
    counts = evaluation.groupby("id")["condition"].nunique()
    complete = counts[counts == len(conditions)].index
    dropped = int(len(counts) - len(complete))
    return evaluation[evaluation["id"].isin(complete)].reset_index(drop=True), dropped


# ------------------------------------------------------------- statistics

def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% by default)."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def confusion_counts(frame: pd.DataFrame) -> dict[str, int]:
    true = frame["true_label"]
    pred = frame["predicted_label"]
    tp = int(((true == "phishing") & (pred == "phishing")).sum())
    tn = int(((true == "legitimate") & (pred == "legitimate")).sum())
    fp = int(((true == "legitimate") & (pred == "phishing")).sum())
    fn = int(((true == "phishing") & (pred != "phishing")).sum())
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": int(len(frame)),
        "n_phishing": int((true == "phishing").sum()),
        "n_legitimate": int((true == "legitimate").sum()),
        "invalid": int((pred == "invalid").sum()),
    }


def classification_metrics(evaluation: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """Accuracy, recall and FPR per condition with Wilson 95% confidence intervals.

    FPR uses all legitimate emails as the denominator, so an invalid answer on a
    legitimate email is not silently dropped (it is a non-error for FPR).
    """
    rows = []
    for condition in conditions:
        counts = confusion_counts(evaluation[evaluation["condition"] == condition])
        specs = [
            ("accuracy", counts["tp"] + counts["tn"], counts["n"]),
            ("phishing_detection_rate", counts["tp"], counts["n_phishing"]),
            ("false_positive_rate", counts["fp"], counts["n_legitimate"]),
        ]
        for metric, k, n in specs:
            low, high = wilson_interval(k, n)
            rows.append({
                "group": "classification", "condition": condition, "metric": metric,
                "value": k / n if n else float("nan"), "ci_low": low, "ci_high": high,
                "k": k, "n": n, "invalid_answers": counts["invalid"],
            })
    return pd.DataFrame(rows)


def paired_tests(evaluation: pd.DataFrame, baseline: str = BASELINE, other: str = PROPOSED) -> pd.DataFrame:
    """Exact McNemar tests for the three hypotheses (same emails under both prompts)."""
    wide_pred = evaluation.pivot(index="id", columns="condition", values="predicted_label")
    truth = evaluation.drop_duplicates("id").set_index("id")["true_label"].reindex(wide_pred.index)

    specs = {
        "accuracy": (truth.index == truth.index, lambda pred: pred == truth),
        "phishing_detection_rate": (truth == "phishing", lambda pred: pred == "phishing"),
        "false_positive_rate": (truth == "legitimate", lambda pred: pred == "phishing"),
    }
    rows = []
    for metric, (mask, indicator) in specs.items():
        subset = wide_pred[np.asarray(mask)]
        a = indicator(wide_pred[baseline])[np.asarray(mask)].astype(bool)
        o = indicator(wide_pred[other])[np.asarray(mask)].astype(bool)
        b = int((a & ~o).sum())
        c = int((~a & o).sum())
        p_raw = mcnemar_exact(b, c)
        rows.append({
            "metric": metric,
            "n_emails": int(len(subset)),
            "baseline_only": b,
            "proposed_only": c,
            "p_value": p_raw,
            "p_value_bonferroni": min(1.0, p_raw * len(specs)),
        })
    return pd.DataFrame(rows)


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, resamples: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    draws = rng.choice(values, size=(resamples, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return (float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5)))


def latency_statistics(evaluation: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """Latency summary per condition (milliseconds)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for condition in conditions:
        values = evaluation.loc[evaluation["condition"] == condition, "latency_ms"].to_numpy(dtype=float)
        low, high = bootstrap_median_ci(values, rng)
        rows.append({
            "group": "latency_ms", "condition": condition, "n": int(len(values)),
            "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
            "median": float(np.median(values)), "median_ci_low": low, "median_ci_high": high,
            "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        })
    return pd.DataFrame(rows)


def latency_by_length(evaluation: pd.DataFrame, conditions: list[str], bins: int = 5) -> pd.DataFrame:
    """Median latency (with bootstrap 95% CI) in quantile bins of email length."""
    rng = np.random.default_rng(SEED)
    lengths = evaluation.drop_duplicates("id").set_index("id")["email_chars"]
    unique_lengths = lengths.nunique()
    n_bins = max(1, min(bins, unique_lengths))
    bin_of = pd.qcut(lengths, q=n_bins, duplicates="drop", labels=False)
    evaluation = evaluation.assign(length_bin=evaluation["id"].map(bin_of))
    rows = []
    for condition in conditions:
        subset = evaluation[evaluation["condition"] == condition]
        for bin_id, group in subset.groupby("length_bin"):
            low, high = bootstrap_median_ci(group["latency_ms"].to_numpy(dtype=float), rng)
            rows.append({
                "condition": condition, "length_bin": int(bin_id), "n": int(len(group)),
                "chars_median": float(group["email_chars"].median()),
                "chars_min": int(group["email_chars"].min()), "chars_max": int(group["email_chars"].max()),
                "latency_median_ms": float(group["latency_ms"].median()),
                "ci_low_ms": low, "ci_high_ms": high,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- reporting

def pct(value: float) -> str:
    return f"{value:.3f}"


def build_results_table(metrics: pd.DataFrame, latency: pd.DataFrame, tests: pd.DataFrame, conditions: list[str]) -> str:
    """Markdown tables for the README 'Results & Discussion' section."""
    names = {
        "accuracy": "Accuracy",
        "phishing_detection_rate": "Phishing detection rate (recall)",
        "false_positive_rate": "False-positive rate",
    }
    header = "| Metric | " + " | ".join(f"{c} (95% CI)" for c in conditions) + " | McNemar p (Bonferroni) |"
    lines = [header, "|" + "---|" * (len(conditions) + 2)]
    for metric, label in names.items():
        cells = []
        for condition in conditions:
            row = metrics[(metrics["condition"] == condition) & (metrics["metric"] == metric)].iloc[0]
            cells.append(f"{pct(row['value'])} [{pct(row['ci_low'])}, {pct(row['ci_high'])}]")
        p = tests[tests["metric"] == metric].iloc[0]["p_value_bonferroni"]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {p:.3f} |")
    lines.append("")
    lines.append("| Latency per request (ms) | n | Median (p25-p75) | p95 | p99 | Mean +/- SD |")
    lines.append("|---|---|---|---|---|---|")
    for condition in conditions:
        r = latency[latency["condition"] == condition].iloc[0]
        lines.append(
            f"| {condition} | {int(r['n'])} | {r['median']:.0f} ({r['p25']:.0f}-{r['p75']:.0f}) | "
            f"{r['p95']:.0f} | {r['p99']:.0f} | {r['mean']:.0f} +/- {r['sd']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def update_readme(readme_path: Path, table_markdown: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise ValueError(f"{readme_path} does not contain the {README_START} / {README_END} markers.")
    before, rest = text.split(README_START, 1)
    _, after = rest.split(README_END, 1)
    readme_path.write_text(f"{before}{README_START}\n\n{table_markdown}\n{README_END}{after}", encoding="utf-8")


def process(raw_path: Path, out_dir: Path) -> dict[str, object]:
    """Run the full processing pipeline and write all tables to out_dir."""
    frame = load_raw(raw_path)
    evaluation, discarded = split_warmup(frame)
    conditions = [c for c in (BASELINE, PROPOSED) if c in set(evaluation["condition"])]
    if len(conditions) < 2:
        raise ValueError(f"Need both '{BASELINE}' and '{PROPOSED}' conditions in the data; found {sorted(set(evaluation['condition']))}.")
    evaluation, incomplete = keep_complete_ids(evaluation, conditions)
    if evaluation.empty:
        raise ValueError("No emails have results under both conditions.")

    metrics = classification_metrics(evaluation, conditions)
    tests = paired_tests(evaluation)
    latency = latency_statistics(evaluation, conditions)
    by_length = latency_by_length(evaluation, conditions)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([metrics, latency], ignore_index=True, sort=False)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    tests.to_csv(out_dir / "paired_tests.csv", index=False)
    by_length.to_csv(out_dir / "latency_by_length.csv", index=False)
    table = build_results_table(metrics, latency, tests, conditions)
    (out_dir / "results_table.md").write_text(table, encoding="utf-8")
    return {
        "evaluation": evaluation, "metrics": metrics, "tests": tests, "latency": latency,
        "by_length": by_length, "table": table, "discarded_warmup": discarded,
        "incomplete_ids_dropped": incomplete, "conditions": conditions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/raw/test_run.jsonl", help="Raw .jsonl log from benchmark.py")
    parser.add_argument("--out", default="data/processed", help="Output directory for aggregated tables")
    parser.add_argument("--update-readme", action="store_true", help="Insert the results table into README.md")
    args = parser.parse_args()
    try:
        result = process(resolve_path(args.raw), resolve_path(args.out))
        if args.update_readme:
            update_readme(PROJECT_ROOT / "README.md", result["table"])
    except (OSError, ValueError) as error:
        print(f"Processing failed: {error}", file=sys.stderr)
        return 1
    print(f"Discarded {result['discarded_warmup']} warm-up records; "
          f"dropped {result['incomplete_ids_dropped']} emails without results in both conditions.")
    print(result["table"])
    print(f"Tables written to: {resolve_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
