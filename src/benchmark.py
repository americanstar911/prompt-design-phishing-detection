"""Compare basic and structured prompts on the same labeled email sample."""

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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


def calculate_metrics(rows: list[dict[str, str]]) -> dict[str, object]:
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


def timed_classify(email_text: str, prompt: str, config: dict) -> tuple[str, str, float]:
    """Classify one email and return (label, raw_response, latency_ms)."""
    start = time.perf_counter()
    predicted, raw = classify(email_text, prompt, config)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return predicted, raw, latency_ms


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def ollama_base_url(chat_url: str) -> str:
    return chat_url.split("/api/", 1)[0]


def collect_environment(config: dict, data_path: Path) -> dict[str, object]:
    """Record software versions and the model digest (best effort, never fails the run)."""
    info: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requests": requests.__version__,
        "model": config["model"],
        "temperature": config["temperature"],
        "seed": config["seed"],
        "dataset_path": str(data_path),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "ollama_version": None,
        "model_digest": None,
    }
    base = ollama_base_url(config["ollama_chat_url"])
    try:
        info["ollama_version"] = requests.get(f"{base}/api/version", timeout=10).json().get("version")
    except Exception:
        pass
    try:
        models = requests.get(f"{base}/api/tags", timeout=10).json().get("models", [])
        wanted = config["model"]
        for item in models:
            name = item.get("name") or item.get("model") or ""
            if name == wanted or name == f"{wanted}:latest":
                info["model_digest"] = item.get("digest")
                break
    except Exception:
        pass
    return info


def load_done_pairs(raw_path: Path) -> set[tuple[str, str]]:
    """Return (condition, id) pairs already stored as evaluation records (for --resume)."""
    done: set[tuple[str, str]] = set()
    if not raw_path.exists():
        return done
    with raw_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("phase") == "eval":
                done.add((record["condition"], record["id"]))
    return done


def run(config_path: Path, resume: bool = False) -> tuple[Path, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = resolve_path(config["dataset_path"])
    rows = load_rows(data_path)
    prompts: dict[str, str] = config["prompts"]
    conditions = list(prompts)
    warmup_requests = int(config.get("warmup_requests", 0))
    raw_path = resolve_path(config.get("raw_output_path", "results/raw_run.jsonl"))
    metadata_path = resolve_path(config.get("metadata_path", str(raw_path.with_name("run_metadata.json"))))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    environment = collect_environment(config, data_path)
    environment["started_utc"] = utc_now()
    environment["warmup_requests_per_condition"] = warmup_requests
    metadata_path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")

    done = load_done_pairs(raw_path) if resume else set()
    mode = "a" if resume and raw_path.exists() else "w"
    total_pairs = len(rows) * len(conditions)

    with raw_path.open(mode, encoding="utf-8") as raw_file:

        def write_record(record: dict) -> None:
            raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw_file.flush()

        # Warm-up: the first requests include model loading and are stored separately.
        if warmup_requests and not (resume and done):
            for index in range(warmup_requests):
                row = rows[index % len(rows)]
                for condition in conditions:
                    predicted, raw, latency_ms = timed_classify(row["email_text"], prompts[condition], config)
                    write_record({
                        "phase": "warmup",
                        "condition": condition,
                        "id": row["id"],
                        "true_label": row["label"],
                        "predicted_label": predicted,
                        "raw_response": raw,
                        "latency_ms": round(latency_ms, 3),
                        "email_chars": len(row["email_text"]),
                        "timestamp_utc": utc_now(),
                    })

        # Evaluation: conditions are interleaved per email, and their order alternates
        # from email to email, so slow drift in speed does not favor one prompt.
        completed = len(done)
        for index, row in enumerate(rows):
            order = conditions if index % 2 == 0 else conditions[::-1]
            for condition in order:
                if (condition, row["id"]) in done:
                    continue
                predicted, raw, latency_ms = timed_classify(row["email_text"], prompts[condition], config)
                write_record({
                    "phase": "eval",
                    "condition": condition,
                    "id": row["id"],
                    "true_label": row["label"],
                    "predicted_label": predicted,
                    "raw_response": raw,
                    "latency_ms": round(latency_ms, 3),
                    "email_chars": len(row["email_text"]),
                    "timestamp_utc": utc_now(),
                })
                completed += 1
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(f"Progress: {completed}/{total_pairs} classifications", file=sys.stderr)

    # Rebuild predictions and metrics from the stored evaluation records.
    records = []
    with raw_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("phase") == "eval":
                    records.append(record)
    order_of_ids = {row["id"]: position for position, row in enumerate(rows)}
    predictions = sorted(
        (
            {
                "condition": r["condition"],
                "id": r["id"],
                "true_label": r["true_label"],
                "predicted_label": r["predicted_label"],
            }
            for r in records
        ),
        key=lambda r: (conditions.index(r["condition"]), order_of_ids[r["id"]]),
    )

    summary = {
        "model": config["model"],
        "dataset": str(data_path.relative_to(PROJECT_ROOT)) if data_path.is_relative_to(PROJECT_ROOT) else str(data_path),
        "temperature": config["temperature"],
        "seed": config["seed"],
        "warmup_requests_per_condition": warmup_requests,
        "conditions": {},
    }
    for condition in conditions:
        condition_rows = [r for r in predictions if r["condition"] == condition]
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

    environment["finished_utc"] = utc_now()
    metadata_path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    return predictions_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.json", help="Path to experiment JSON config")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run: skip emails already stored in the raw .jsonl file",
    )
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    try:
        predictions_path, summary_path = run(config_path, resume=args.resume)
    except requests.ConnectionError:
        print("Could not reach Ollama. Open the Ollama app and make sure the local server is running.", file=sys.stderr)
        return 1
    except requests.Timeout:
        print("Ollama request timed out. Try again or increase timeout_seconds in the config.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Finished records are saved; rerun with --resume to continue.", file=sys.stderr)
        return 130
    except (KeyError, ValueError, json.JSONDecodeError, requests.RequestException) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    print(f"Predictions saved to: {predictions_path}")
    print(f"Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
