# Domain Model (Software Engineering track)

This document describes the benchmark system as a C4 model and marks the **metric
collection points** (M1-M5): the places in the code where measurements are taken.
Diagrams are written in [Mermaid](https://mermaid.js.org/) and render on GitHub.

## 1. System context (C4 level 1)

```mermaid
flowchart LR
    researcher(["Researcher<br/>runs the experiment,<br/>reads the results"])
    system["Prompt-design benchmark<br/>[software system]<br/>compares a basic and a structured prompt<br/>on the same labeled emails"]
    ollama[("Ollama + llama3.2:3b<br/>[external system, local]<br/>classifies one email per request")]
    dataset[("Public phishing-email dataset<br/>[external data, CSV]")]

    researcher -->|"configures, starts, reads reports"| system
    system -->|"HTTP POST /api/chat"| ollama
    dataset -->|"downloaded once"| system
```

## 2. Containers and metric collection points (C4 level 2)

```mermaid
flowchart TB
    researcher(["Researcher"])
    dataset[("Phishing_Email.csv<br/>data/external, not committed")]

    subgraph repo["Repository: prompt-design-phishing-detection"]
        direction TB
        prep["prepare_dataset.py<br/>[Python script]<br/>cleans, de-duplicates, samples<br/>(seed 42)"]
        sample[("emails_eval.csv<br/>data/sample<br/>balanced labeled sample")]
        config[/"config.json / config_eval.json<br/>model, prompts, temperature, seed"/]
        bench["benchmark.py<br/>[Python script]<br/>runs both prompts on every email"]
        raw[("test_run.jsonl<br/>data/raw<br/>one record per request,<br/>warm-up and evaluation")]
        meta[("run_metadata.json<br/>versions, model digest,<br/>dataset hash")]
        proc["process_data.py<br/>[Python module]<br/>drops warm-up, computes metrics,<br/>95% CI, McNemar tests"]
        processed[("metrics_summary.csv<br/>paired_tests.csv<br/>data/processed")]
        viz["visualize.py<br/>[Python module]<br/>figures as PDF + PNG 300 DPI"]
        figs[("reports/figures")]
        nb["01_data_analysis.ipynb<br/>[Jupyter notebook]<br/>reproducible analysis"]
        M1(("M1"))
        M2(("M2"))
        M3(("M3"))
        M4(("M4"))
        M5(("M5"))
    end

    ollama[("Ollama server<br/>localhost:11434")]

    researcher --> prep
    dataset --> prep
    prep --> sample
    sample --> bench
    config --> bench
    bench -->|"POST /api/chat"| ollama
    ollama -->|"JSON answer"| bench
    bench --> raw
    bench --> meta
    bench -.-> M1
    bench -.-> M2
    bench -.-> M3
    bench -.-> M4
    bench -.-> M5
    raw --> proc
    proc --> processed
    processed --> viz
    proc --> viz
    viz --> figs
    nb -. "calls" .-> proc
    nb -. "calls" .-> viz

    classDef metric fill:#fde8d0,stroke:#d55e00,stroke-width:2px,color:#000
    class M1,M2,M3,M4,M5 metric
```

### Metric collection points

| ID | Where (function in `src/benchmark.py`) | What is measured | Stored as |
|---|---|---|---|
| **M1** | `timed_classify` wraps the HTTP request to Ollama with `time.perf_counter()` | Latency of one classification, in milliseconds | `latency_ms` in `test_run.jsonl` |
| **M2** | `classify` and `parse_label` | Raw model answer and the label parsed from it (`phishing`, `legitimate`, or `invalid`) | `raw_response`, `predicted_label` |
| **M3** | `run` (`len(email_text)`) | Size of the input email in characters | `email_chars` |
| **M4** | `collect_environment` | Python and platform, Ollama version, model digest, temperature, seed, SHA-256 of the dataset file | `run_metadata.json` |
| **M5** | `run` (`utc_now`) | Time of every request, plus the `phase` (`warmup` or `eval`) | `timestamp_utc`, `phase` |

Warm-up requests (`phase = warmup`) include loading the model into memory and are
stored but excluded from all statistics by `process_data.split_warmup`.

## 3. Components of `benchmark.py` (C4 level 3)

```mermaid
flowchart LR
    main["main()<br/>CLI, error handling"] --> run["run()<br/>orchestration"]
    run --> load["load_rows()<br/>validate CSV and labels"]
    run --> env["collect_environment()<br/>[M4]"]
    run --> timed["timed_classify()<br/>[M1]"]
    timed --> classify["classify()<br/>builds request, calls Ollama<br/>[M2]"]
    classify --> parse["parse_label()<br/>JSON, then regex, else invalid<br/>[M2]"]
    run --> metrics["calculate_metrics()<br/>TP, TN, FP, FN, rates"]
    run --> out["writes jsonl, predictions.csv,<br/>summary.json, run_metadata.json"]
```

Design decisions that protect validity:

- **Same inputs for both prompts.** Every email is sent under both conditions; only the user prompt differs.
- **Interleaving and alternating order.** Conditions are interleaved per email and their order alternates, so slow drift in machine speed does not favor one prompt.
- **Isolated requests.** Each request is stateless: no chat history is shared between emails.
- **Incremental writes.** Every record is flushed to disk immediately, and `--resume` continues an interrupted run without repeating finished emails.

## 4. Data model of one raw record

| Field | Type | Meaning |
|---|---|---|
| `phase` | string | `warmup` or `eval` |
| `condition` | string | `basic` or `structured` |
| `id` | string | Email identifier from the dataset sample |
| `true_label` | string | Ground truth: `phishing` or `legitimate` |
| `predicted_label` | string | Model decision: `phishing`, `legitimate`, or `invalid` |
| `raw_response` | string | Unmodified model output |
| `latency_ms` | number | Request duration in milliseconds |
| `email_chars` | integer | Email length in characters |
| `timestamp_utc` | string | ISO 8601 time of the request |

## 5. Threats to validity (summary)

| Threat | Mitigation in this design |
|---|---|
| Latency affected by model loading | Warm-up records are separated and discarded |
| Order effects (thermal throttling, caching) | Conditions interleaved, order alternated |
| Random generation | `temperature = 0`, `seed = 42` |
| Unclear software state | Versions, model digest, and dataset hash recorded (M4) |
| Ambiguous labels in the dataset | Texts that carry both labels are removed in `prepare_dataset.py` |
| One small model, one dataset | Stated as a limitation; results are not generalized |
