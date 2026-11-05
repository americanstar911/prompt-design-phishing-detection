The Effect of Prompt Design on AI-Based Phishing Email Detection

Author: Orazgeldy Nurdaulet
Research type: Quantitative comparative experiment
License: MIT

Overview

This project compares a basic prompt and a structured prompt for classifying emails as phishing or legitimate. Both prompts are tested with the same local language model and the same labeled emails.

Research question: Does prompt design affect the accuracy, phishing detection rate, and false-positive rate of an AI model classifying emails?

Hypothesis: The classification results will differ between the basic and structured prompt conditions. The null hypothesis is that the results will not differ.

Short answer: Yes, for accuracy and phishing detection. On 400 emails from a public dataset, the more detailed structured prompt found fewer phishing emails than the basic prompt (see Results & Discussion).

Repository structure
text
configs/config.json         Smoke-test settings (8 synthetic emails) and prompt definitions
configs/config_eval.json    Settings for the final experiment (balanced public-dataset sample)
data/sample/emails.csv      Eight synthetic emails for a smoke test
data/sample/emails_eval.csv Balanced evaluation sample drawn from a public dataset (created by prepare_dataset.py)
data/external/              Downloaded public dataset (not committed)
data/raw/test_run.jsonl     Raw benchmark log: one record per request, warm-up and evaluation
data/processed/             Aggregated metrics and statistical tests (CSV)
docs/methodology_card.md    Methodology passport card
docs/domain_model.md        C4 domain model with metric collection points
notebooks/01_data_analysis.ipynb  Reproducible analysis notebook
reports/figures/            Figures as PDF (vector) and PNG (300 DPI)
src/benchmark.py            Runs both prompt conditions, times every request, writes the raw log
src/prepare_dataset.py      Builds the balanced, reproducible evaluation sample
src/process_data.py         Drops warm-up, computes metrics, confidence intervals, McNemar tests
src/visualize.py            Publication-ready figures
tests/                      Unit and end-to-end tests
environment.yml             Pinned Conda/Python environment
requirements.txt            Pinned Python packages
System requirements
macOS, Windows, or Linux
Python 3.9 or later (the pinned libraries support Python 3.9 to 3.12)
Ollama installed and running locally
About 2 GB free disk space for the llama3.2:3b model

The model is served locally by Ollama. No API key is required. The model tag and generation settings are recorded in the configuration.

The reproducible Conda environment pins Python 3.12.8 and the Python dependencies. Create it with conda env create -f environment.yml and activate it with conda activate phishing-prompt-benchmark.

Quickstart (smoke test)
Install Ollama from ollama.com/download, open it, then download the model:
bash
ollama pull llama3.2:3b
In the repository folder, create and activate a Python environment and install the pinned dependencies:
bash
python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt
Run the sample benchmark:
bash
python src/benchmark.py --config configs/config.json

The script saves results/predictions.csv, results/summary.json, and a raw log with per-request latency. The 8-email sample output is a pipeline check only; it is not a research result. The real results are in the Results section below.

Run the unit tests with python -m unittest discover -s tests. The end-to-end tests use a fake local server, so they do not need Ollama or a model.

Running the final experiment
Get the public dataset. Download Phishing_Email.csv from the zefang-liu/phishing-email-dataset page (a copy of the Kaggle dataset "Phishing Email Detection"; the dataset card states the LGPL-3.0 license) and save it as data/external/Phishing_Email.csv.
Create the evaluation sample. This removes empty, very short, duplicate, and conflicting-label texts, draws a class-balanced sample (200 phishing + 200 legitimate) with seed 42, and truncates long emails to 2,000 characters:
bash
python src/prepare_dataset.py --input data/external/Phishing_Email.csv --output data/sample/emails_eval.csv --per-class 200 --seed 42 --max-chars 2000
Run the benchmark (3 warm-up requests per condition, then every email under both prompts). Finished records are saved immediately; if the run is interrupted, continue it with --resume:
bash
python src/benchmark.py --config configs/config_eval.json
python src/benchmark.py --config configs/config_eval.json --resume
Process the raw log and build figures, then update the results table below:
bash
python src/process_data.py --raw data/raw/test_run.jsonl --update-readme
python src/visualize.py --raw data/raw/test_run.jsonl
Open the notebook notebooks/01_data_analysis.ipynb and run Kernel -> Restart & Run All so that all outputs are saved in the file.
Metrics
Metric	Role	Formula
Phishing detection rate (recall)	Primary	TP / (TP + FN)
Accuracy	Guardrail	(TP + TN) / N
False-positive rate	Guardrail	FP / (FP + TN)

In plain words:

Recall answers: of all real phishing emails, how many did the model catch? This is the main metric, because a missed phishing email is the most dangerous mistake.
Accuracy answers: how many of all answers were correct? It checks that the overall quality did not drop.
False-positive rate answers: how many normal emails were wrongly called phishing? It checks that catching more phishing does not come from blocking normal mail.

The basic prompt is the baseline. Both conditions are evaluated on the same emails. Proportions are reported with Wilson 95% confidence intervals. Because both prompts see the same emails, the prompts are compared with an exact McNemar test, which only looks at emails where the two prompts disagree. A Bonferroni correction is applied because three hypotheses are tested. Request latency is reported as median (p25-p75), p95, p99 and mean +/- SD after discarding warm-up requests.

Results & Discussion

The table below is generated by python src/process_data.py --update-readme from the raw log. Do not edit it by hand; the text between the markers is overwritten on every update.

<!-- RESULTS:START -->
Metric	basic (95% CI)	structured (95% CI)	McNemar p (Bonferroni)
Accuracy	0.805 [0.763, 0.841]	0.757 [0.713, 0.797]	0.013
Phishing detection rate (recall)	0.660 [0.592, 0.722]	0.530 [0.461, 0.598]	0.000
False-positive rate	0.050 [0.027, 0.090]	0.015 [0.005, 0.043]	0.117
Latency per request (ms)	n	Median (p25-p75)	p95	p99	Mean +/- SD
basic	400	1109 (715-1887)	2909	3815	1376 +/- 891
structured	400	1163 (749-1886)	2954	3628	1420 +/- 924
<!-- RESULTS:END -->
Discussion

We tested 400 emails (200 phishing, 200 legitimate). The more detailed "structured" prompt did worse than the simple "basic" prompt. It found fewer phishing emails (recall 0.530 vs 0.660) and was less accurate overall (0.757 vs 0.805). Both differences are statistically significant even after the Bonferroni correction (exact McNemar test, p = 0.013 for accuracy and p < 0.001 for recall), so we reject the null hypotheses for RQ1 (accuracy) and RQ2 (phishing detection).

The false-positive rate was lower with the structured prompt (0.015 vs 0.050), but this difference could be chance (adjusted p = 0.117), so we cannot reject the null hypothesis for RQ3.

What this suggests: the longer prompt made the small model more cautious. It said "phishing" less often, which removed some false alarms but let more real phishing through. This is the opposite of the common expectation that a more detailed prompt is always better. Speed was almost the same (median 1109 ms vs 1163 ms).

These results are for one small model (llama3.2:3b), one dataset sample, and emails cut to 2,000 characters. They should not be applied to other models or datasets without testing.

Figures

Median latency vs email load

Latency distribution per prompt

Accuracy, recall and false-positive rate with 95% CI

Vector versions of all figures are saved next to the PNG files as PDF.

Domain model

The C4 model of the benchmark system, with the metric collection points M1-M5, is in docs/domain_model.md.

Limitations
One small local model (llama3.2:3b) and one public dataset: results do not automatically generalize to other models, languages, or phishing types.
Long emails are truncated to 2,000 characters, so the model sees at most that much text.
Each prompt was run once per email. With temperature 0 and a fixed seed the output does not vary, so run-to-run variation cannot be measured.
Latency is measured on one machine and depends on its hardware; only the comparison between the two prompts, measured in interleaved order, is meaningful.
An invalid model answer on a phishing email counts as a missed phishing email. In the confidence intervals and tests an invalid answer on a legitimate email counts as not a false positive, and the number of invalid answers is reported with every result.
Record the Ollama version and model digest (saved automatically in data/raw/run_metadata.json) when reporting results.
Citation and license

The code is released under the MIT License. The messages in data/sample/emails.csv are synthetic. data/sample/emails_eval.csv is a sample derived from the public dataset named above; follow that dataset's license and attribution terms when reusing it. Cite this project as:

Nurdaulet, O. The Effect of Prompt Design on AI-Based Phishing
