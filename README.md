# The Effect of Prompt Design on AI-Based Phishing Email Detection

**Author:** Orazgeldy Nurdaulet  
**Research type:** Quantitative comparative experiment  
**License:** MIT

## Overview

This project compares a basic prompt and a structured prompt for classifying emails as phishing or legitimate. Both prompts are tested with the same local language model and the same labeled emails.

**Research question:** Does prompt design affect the accuracy, phishing detection rate, and false-positive rate of an AI model classifying emails?

**Hypothesis:** The classification results will differ between the basic and structured prompt conditions. The null hypothesis is that the results will not differ.

## Repository structure

```text
configs/config.json       Experiment settings and prompt definitions
data/sample/emails.csv    Eight synthetic emails for a smoke test
docs/methodology_card.md  Methodology passport card
src/benchmark.py          Runs both prompt conditions and calculates metrics
tests/test_benchmark.py   Checks label parsing and metric calculations
environment.yml           Pinned Conda/Python environment
requirements.txt          Pinned Python package dependency
```

## System requirements

- macOS, Windows, or Linux
- Python 3.10 or later
- Ollama installed and running locally
- About 2 GB free disk space for the `llama3.2:3b` model

The model is served locally by Ollama. No API key is required. The model tag and generation settings are recorded in the configuration.

The reproducible Conda environment pins Python 3.12.8 and the Python dependency. Create it with `conda env create -f environment.yml` and activate it with `conda activate phishing-prompt-benchmark`.

## Quickstart

1. Install Ollama from [ollama.com/download](https://ollama.com/download), open it, then download the model:

```bash
ollama pull llama3.2:3b
```

2. In the repository folder, create and activate a Python environment and install the pinned dependency:

```bash
python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt
```

3. Run the sample benchmark:

```bash
python src/benchmark.py --config configs/config.json
```

The script saves `results/predictions.csv` and `results/summary.json`. The sample output is a pipeline check only; it is not a final research result.

Run the lightweight unit tests with `python -m unittest discover -s tests`.

## Planned benchmark metrics

| Metric | Role | Formula |
|---|---|---|
| Phishing detection rate (recall) | Primary | TP / (TP + FN) |
| Accuracy | Guardrail | (TP + TN) / N |
| False-positive rate | Guardrail | FP / (FP + TN) |

The basic prompt is the baseline. The final study should use a larger labeled public email dataset and report both conditions on the same examples.

## Citation and license

The code is released under the MIT License. The included sample messages are synthetic. Cite this project as:

```text
Nurdaulet, O. The Effect of Prompt Design on AI-Based Phishing Email Detection. GitHub repository, 2026.
```
