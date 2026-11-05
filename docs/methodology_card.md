# Methodology Passport Card

## Project information

- **Title:** The Effect of Prompt Design on AI-Based Phishing Email Detection
- **Author:** Orazgeldy Nurdaulet
- **GitHub repository:** https://github.com/americanstar911/prompt-design-phishing-detection

## Research type

This is a quantitative comparative experiment. It measures how two prompt designs affect the classification of the same labeled emails by one fixed language model.

## Research questions and hypotheses

**RQ1:** Does prompt design affect overall classification accuracy?

- **H0₁:** Accuracy is equal for the basic and structured prompts.
- **H1₁:** Accuracy differs between the basic and structured prompts.

**RQ2:** Does prompt design affect the phishing detection rate?

- **H0₂:** The phishing detection rate is equal for both prompts.
- **H1₂:** The phishing detection rate differs between the prompts.

**RQ3:** Does prompt design affect the false-positive rate on legitimate emails?

- **H0₃:** The false-positive rate is equal for both prompts.
- **H1₃:** The false-positive rate differs between the prompts.

## Variable matrix

| Variable type | Definition |
|---|---|
| Independent variable | Prompt design with two conditions: basic and structured. |
| Dependent variables | Accuracy, phishing detection rate (recall), and false-positive rate. |
| Controlled variables | Model and model tag, labeled email set, email order, system instruction, output format, temperature, seed, timeout, and evaluation code. |

## Metrics and baseline

- **Primary metric:** Phishing detection rate, TP / (TP + FN).
- **Guardrail metrics:** Accuracy, (TP + TN) / N; and false-positive rate, FP / (FP + TN).
- **Baseline:** Basic prompt. The structured prompt is compared against it.
- **Definitions:** TP = phishing correctly labeled phishing; FN = phishing labeled legitimate or invalid; FP = legitimate labeled phishing; TN = legitimate correctly labeled legitimate.

The 8 synthetic emails in `data/sample/emails.csv` are only a pipeline smoke test. They are not sufficient to support research conclusions. A larger, documented, labeled public dataset must be selected for the final experiment.

## Dataset

The final experiment uses a public labeled email dataset: zefang-liu/phishing-email-dataset, file Phishing_Email.csv (https://huggingface.co/datasets/zefang-liu/phishing-email-dataset).
From it, 400 emails were sampled: 200 phishing and 200 legitimate, so the two classes are balanced.
The sample was drawn with random seed 42, so anyone can get the same 400 emails.
Each email was cut to 2,000 characters to keep requests short and the runtime reasonable.
The sample is saved in `data/sample/emails_eval.csv`, with its description in `emails_eval.meta.json`.

The 8 synthetic emails in `data/sample/emails.csv` are kept only as a quick test that the code runs.

## Experimental procedure

Both prompts (basic and structured) were run on all 400 emails with the same local model,
llama3.2:3b, through Ollama. Temperature was 0 and the seed was 42, so the model gives the same
answer to the same input. Each email was sent as a separate request, so one email cannot affect another.
Before the main run, 6 warm-up requests were made and not counted. The main run has 800 requests (400 emails x 2 prompts).
The three metrics were calculated separately for each prompt.

To check whether a difference is real or just chance, we used the exact McNemar test.
It compares two prompts on the same emails and only looks at emails where the prompts disagree.
Because we test three hypotheses, p-values were adjusted with the Bonferroni correction.

## Limitations

- Only one small model (llama3.2:3b) was tested. Larger models may behave differently.
- Only one dataset sample was used (400 emails, cut to 2,000 characters). Results may differ for other datasets, languages, or phishing types.
- Each prompt was run once. With temperature 0 and a fixed seed the output does not vary, so we cannot measure run-to-run variation.
- If the model gives an answer in the wrong format, it is marked "invalid". An invalid answer on a phishing email counts as a miss.
- Ollama and the model version must be recorded for reproducibility.