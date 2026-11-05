# The Effect of Prompt Design on AI-Based Phishing Email Detection

**Author:** Orazgeldy Nurdaulet  
**Repository:** https://github.com/americanstar911/prompt-design-phishing-detection (tag: task3)

## 1. Goal

We checked whether the way a prompt is written changes how well an AI model finds phishing emails.

## 2. Method

- **Model:** llama3.2:3b, run locally with Ollama (temperature 0, seed 42), so the model gives the same answer to the same input.
- **Data:** 400 emails (200 phishing, 200 legitimate) sampled with seed 42 from the public dataset [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) (`Phishing_Email.csv`). Empty, very short, duplicate and conflicting-label texts were removed first. Each email was cut to 2,000 characters.
- **Two prompts:** *basic* (just classify the email as phishing or legitimate) and *structured* (also asks the model to check the sender, links or attachments, requests for passwords or money, and urgency, and to consider these signs together).
- **Same for both prompts:** model, emails, order, system instruction, output format (JSON), temperature, seed and evaluation code. Only the prompt changes.
- **Metrics:** recall (share of phishing emails found; main metric), accuracy (share of correct answers), false-positive rate (share of normal emails wrongly called phishing).
- **Statistics:** exact McNemar test (compares two prompts on the same emails), with a Bonferroni correction because three hypotheses are tested. Proportions come with 95% Wilson confidence intervals.

## 3. Results

| Metric | Basic | Structured | Adjusted p | Conclusion |
|---|---|---|---|---|
| Accuracy | 0.805 | 0.757 | 0.013 | Difference is significant |
| Recall (phishing found) | 0.660 | 0.530 | < 0.001 | Difference is significant |
| False-positive rate | 0.050 | 0.015 | 0.117 | Not significant |
| Median latency | 1109 ms | 1163 ms | - | Almost the same |

Figures are in `reports/figures/`:

- `fig1_latency_vs_load` - median latency vs email load
- `fig2_distribution_boxplot` - latency distribution per prompt
- `fig3_metrics_with_ci` - accuracy, recall and false-positive rate with 95% confidence intervals

## 4. Discussion

The detailed prompt made the small model more cautious. It said "phishing" less often, so it had fewer false alarms but missed more real phishing emails. In this setup, a longer and more detailed prompt did not help; it made the main metric (recall) worse. We reject the null hypotheses for accuracy (RQ1) and phishing detection (RQ2). For the false-positive rate (RQ3) we cannot reject the null hypothesis, because the difference could be chance.

## 5. Limitations

- Only one small model (llama3.2:3b) and one dataset sample were tested, so the results should not be applied to other models, languages or phishing types.
- Emails were cut to 2,000 characters.
- Each prompt was run once per email. With temperature 0 and a fixed seed the output does not vary, so run-to-run variation cannot be measured.
- An invalid model answer on a phishing email counts as a missed phishing email.
- Latency was measured on one machine, so only the comparison between the two prompts is meaningful.

## 6. Reproducibility

Code, data sample, raw log, environment files and instructions are in the repository. After installing Ollama and running `ollama pull llama3.2:3b`, the experiment is run with `python src/benchmark.py --config configs/config_eval.json`. The domain model (C4) with metric collection points is in `docs/domain_model.md`.
