"""Build a reproducible, class-balanced evaluation sample from a public phishing-email CSV.

Example:
    python src/prepare_dataset.py --input data/external/Phishing_Email.csv \
        --output data/sample/emails_eval.csv --per-class 200 --seed 42 --max-chars 2000
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_COLUMNS = ["email text", "email_text", "text", "body", "content", "message"]
LABEL_COLUMNS = ["email type", "email_type", "label", "class", "type", "category"]
PHISHING_VALUES = {"phishing email", "phishing", "phish"}
LEGITIMATE_VALUES = {"safe email", "legitimate", "ham", "safe", "benign"}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_column(columns, candidates, kind: str) -> str:
    lowered = {str(c).strip().lower(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    raise ValueError(f"Could not find the {kind} column. Columns in file: {list(columns)}. Expected one of: {candidates}")


def normalise_label(value) -> str:
    text = str(value).strip().lower()
    if text in PHISHING_VALUES:
        return "phishing"
    if text in LEGITIMATE_VALUES:
        return "legitimate"
    raise ValueError(f"Unknown label value: {value!r}")


def build_sample(
    frame: pd.DataFrame,
    per_class: int,
    seed: int,
    max_chars: int,
    min_chars: int,
) -> tuple[pd.DataFrame, dict]:
    text_col = find_column(frame.columns, TEXT_COLUMNS, "email text")
    label_col = find_column(frame.columns, LABEL_COLUMNS, "label")

    stats: dict = {"rows_in_file": int(len(frame)), "text_column": str(text_col), "label_column": str(label_col)}
    known = PHISHING_VALUES | LEGITIMATE_VALUES
    unknown = sorted(str(v) for v in frame[label_col].dropna().unique() if str(v).strip().lower() not in known)
    if unknown:
        raise ValueError(f"Unknown label values in file: {unknown}. Update PHISHING_VALUES / LEGITIMATE_VALUES.")

    data = pd.DataFrame({"label": frame[label_col].map(normalise_label), "email_text": frame[text_col]})
    data = data.dropna(subset=["email_text"])
    data["email_text"] = data["email_text"].astype(str).str.replace("\x00", "", regex=False).str.strip()
    stats["after_dropping_empty"] = int(len(data))

    data = data[data["email_text"].str.len() >= min_chars]
    stats["after_min_length"] = int(len(data))

    # An identical text carrying both labels is ambiguous ground truth: drop all its copies.
    labels_per_text = data.groupby("email_text")["label"].nunique()
    conflicting = labels_per_text[labels_per_text > 1].index
    stats["texts_with_conflicting_labels_removed"] = int(len(conflicting))
    data = data[~data["email_text"].isin(conflicting)]

    data = data.drop_duplicates(subset=["email_text"], keep="first")
    stats["after_deduplication"] = int(len(data))

    available = data["label"].value_counts().to_dict()
    stats["available_per_class"] = {k: int(v) for k, v in available.items()}
    for label in ("phishing", "legitimate"):
        if available.get(label, 0) < per_class:
            raise ValueError(f"Only {available.get(label, 0)} '{label}' emails available, but --per-class is {per_class}.")

    sample = data.groupby("label", group_keys=False).sample(n=per_class, random_state=seed)
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)

    stats["original_chars_median"] = float(sample["email_text"].str.len().median())
    stats["original_chars_max"] = int(sample["email_text"].str.len().max())
    if max_chars > 0:
        stats["truncated_emails"] = int((sample["email_text"].str.len() > max_chars).sum())
        sample["email_text"] = sample["email_text"].str.slice(0, max_chars)
    else:
        stats["truncated_emails"] = 0

    sample.insert(0, "id", [f"E{i + 1:04d}" for i in range(len(sample))])
    stats["sample_size"] = int(len(sample))
    stats["sample_per_class"] = {k: int(v) for k, v in sample["label"].value_counts().to_dict().items()}
    stats["sample_chars_median"] = float(sample["email_text"].str.len().median())
    stats["sample_chars_max"] = int(sample["email_text"].str.len().max())
    return sample[["id", "label", "email_text"]], stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the downloaded public CSV (e.g. Phishing_Email.csv)")
    parser.add_argument("--output", default="data/sample/emails_eval.csv", help="Where to write the evaluation sample")
    parser.add_argument("--per-class", type=int, default=200, help="Emails per class (default 200 -> 400 total)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--max-chars", type=int, default=2000, help="Truncate emails to this many characters (0 = no limit)")
    parser.add_argument("--min-chars", type=int, default=20, help="Drop emails shorter than this")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    try:
        frame = pd.read_csv(input_path)
        sample, stats = build_sample(frame, args.per_class, args.seed, args.max_chars, args.min_chars)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False, encoding="utf-8")
    meta = {
        "source_file": input_path.name,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "seed": args.seed,
        "per_class": args.per_class,
        "max_chars": args.max_chars,
        "min_chars": args.min_chars,
        "statistics": stats,
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Sample saved to: {output_path}")
    print(f"Sampling details saved to: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
