"""Compare basic and structured prompts on the same labeled email sample."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {"phishing", "legitimate"}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"id", "label", "email_text"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CSV must contain id, label, and email_text columns and at least one row.")
    for row in rows:
        row["label"] = row["label"].strip().lower()
        if row["label"] not in LABELS:
            raise ValueError(f"Invalid label for {row['id']}: {row['label']}")
    return rows


def parse_label(content: str) -> str:
    try:
        value = json.loads(content).get("label", "")
        label = str(value).strip().lower()
        if label in LABELS:
            return label
    except (json.JSONDecodeError, AttributeError):
        pass
    match = re.search(r"\b(phishing|legitimate)\b", content.lower())
    return match.group(1) if match else "invalid"


def classify(email_text: str, prompt: str, config: dict) -> tuple[str, str]:
    user_prompt = prompt.replace("{email}", email_text)
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": config["temperature"], "seed": config["seed"]},
    }
    response = requests.post(
        config["ollama_chat_url"],
        json=payload,
        timeout=config["timeout_seconds"],
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    return parse_label(content), content


def calculate_metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    tp = sum(r["true_label"] == "phishing" and r["predicted_label"] == "phishing" for r in rows)
    tn = sum(r["true_label"] == "legitimate" and r["predicted_label"] == "legitimate" for r in rows)
    fp = sum(r["true_label"] == "legitimate" and r["predicted_label"] == "phishing" for r in rows)
    fn = sum(r["true_label"] == "phishing" and r["predicted_label"] != "phishing" for r in rows)
    total = len(rows)
    phishing_total = tp + fn
    legitimate_total = fp + tn
    return {
        "n": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "phishing_detection_rate": tp / phishing_total if phishing_total else 0.0,
        "false_positive_rate": fp / legitimate_total if legitimate_total else 0.0,
        "invalid_output_count": sum(r["predicted_label"] == "invalid" for r in rows),
    }


def run(config_path: Path) -> tuple[Path, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = resolve_path(config["dataset_path"])
    rows = load_rows(data_path)
    predictions: list[dict[str, str]] = []
    summary = {
        "model": config["model"],
        "dataset": str(data_path.relative_to(PROJECT_ROOT)),
        "temperature": config["temperature"],
        "seed": config["seed"],
        "conditions": {},
    }

    for condition, prompt in config["prompts"].items():
        condition_rows = []
        for row in rows:
            predicted, raw = classify(row["email_text"], prompt, config)
            result = {
                "condition": condition,
                "id": row["id"],
                "true_label": row["label"],
                "predicted_label": predicted,
            }
            predictions.append(result)
            condition_rows.append(result)
        summary["conditions"][condition] = calculate_metrics(condition_rows)

    results_dir = resolve_path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = results_dir / "predictions.csv"
    summary_path = results_dir / "summary.json"
    with predictions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["condition", "id", "true_label", "predicted_label"])
        writer.writeheader()
        writer.writerows(predictions)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return predictions_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.json", help="Path to experiment JSON config")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    try:
        predictions_path, summary_path = run(config_path)
    except requests.ConnectionError:
        print("Could not reach Ollama. Open the Ollama app and make sure the local server is running.", file=sys.stderr)
        return 1
    except requests.Timeout:
        print("Ollama request timed out. Try again or increase timeout_seconds in the config.", file=sys.stderr)
        return 1
    except (KeyError, ValueError, json.JSONDecodeError, requests.RequestException) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    print(f"Predictions saved to: {predictions_path}")
    print(f"Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
