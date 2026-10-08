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

## Experimental procedure

Run both prompt conditions on every email using the same local `llama3.2:3b` model through Ollama. Keep the listed settings fixed, save each predicted label, and calculate all three metrics separately for each prompt. Record the Ollama model digest and software versions when collecting final results.

## Limitations

The sample is small and synthetic. Results from one model and one dataset will not automatically generalize to other models, languages, or phishing types. Ollama and the model version must be recorded for reproducibility.
