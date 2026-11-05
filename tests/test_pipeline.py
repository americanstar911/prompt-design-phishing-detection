"""End-to-end tests of the benchmark, processing and plotting code.

A tiny fake Ollama server answers by keyword rules. These results only exercise the
code paths; they are NOT experimental data.
"""

import csv
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import benchmark  # noqa: E402
import prepare_dataset  # noqa: E402
import process_data  # noqa: E402
import visualize  # noqa: E402
import pandas as pd  # noqa: E402


class FakeOllama(BaseHTTPRequestHandler):
    """Classifies by keywords; the 'structured' prompt misses one specific email."""

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._send({"version": "0.0.test"})
        elif self.path == "/api/tags":
            self._send({"models": [{"name": "fake-model:latest", "digest": "abc123"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        user = payload["messages"][-1]["content"].lower()
        time.sleep(0.001 + len(user) / 400000)
        email = user.split("email:\n", 1)[-1]  # judge the email only, not the instruction text
        is_phish = any(word in email for word in ("link", "verify", "fee", "reply with"))
        structured = "check the sender identity" in user
        if structured and "redelivery" in email:
            is_phish = False
        label = "phishing" if is_phish else "legitimate"
        self._send({"message": {"content": json.dumps({"label": label})}})


SAMPLE_ROWS = [
    ("S01", "phishing", "Your account will be suspended today. Use the link below to verify your account."),
    ("S02", "phishing", "A package delivery is on hold. Pay the small redelivery fee using this link."),
    ("S03", "phishing", "Your mailbox is full. Sign in now using this link to prevent blocked messages."),
    ("S04", "phishing", "An unusual payment was detected. Reply with your account details to cancel it."),
    ("S05", "legitimate", "The library will close at 6 pm today. Please return borrowed books by Friday."),
    ("S06", "legitimate", "Your course assignment is due next Monday. The rubric is in the student portal."),
    ("S07", "legitimate", "The department meeting has moved to Room 204. The time remains 2 pm."),
    ("S08", "legitimate", "Your order has shipped. You can check its status in the store app you use."),
]


class ProcessingMath(unittest.TestCase):
    def test_wilson_interval_zero_successes(self):
        low, high = process_data.wilson_interval(0, 8)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.3244, places=3)

    def test_wilson_interval_bounds(self):
        low, high = process_data.wilson_interval(8, 8)
        self.assertAlmostEqual(low, 0.6756, places=3)
        self.assertEqual(high, 1.0)

    def test_mcnemar_exact(self):
        self.assertEqual(process_data.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(process_data.mcnemar_exact(1, 0), 1.0)
        self.assertAlmostEqual(process_data.mcnemar_exact(5, 0), 0.0625)
        self.assertAlmostEqual(process_data.mcnemar_exact(3, 3), 1.0)

    def test_confusion_counts_treat_invalid_phishing_as_miss(self):
        frame = pd.DataFrame({
            "true_label": ["phishing", "phishing", "legitimate", "legitimate"],
            "predicted_label": ["phishing", "invalid", "legitimate", "phishing"],
        })
        counts = process_data.confusion_counts(frame)
        self.assertEqual((counts["tp"], counts["fn"], counts["tn"], counts["fp"]), (1, 1, 1, 1))
        self.assertEqual(counts["invalid"], 1)


class DatasetPreparation(unittest.TestCase):
    def test_build_sample_balances_and_truncates(self):
        rows = []
        for i in range(30):
            rows.append(("Phishing Email", f"verify your account now number {i} " * 5))
            rows.append(("Safe Email", f"meeting notes for project number {i} " * 5))
        rows.append(("Safe Email", None))
        rows.append(("Safe Email", "short"))
        rows.append(("Phishing Email", rows[0][1]))  # exact duplicate
        frame = pd.DataFrame(rows, columns=["Email Type", "Email Text"])
        sample, stats = prepare_dataset.build_sample(frame, per_class=10, seed=42, max_chars=60, min_chars=20)
        self.assertEqual(len(sample), 20)
        self.assertEqual(sample["label"].value_counts().to_dict(), {"phishing": 10, "legitimate": 10})
        self.assertTrue((sample["email_text"].str.len() <= 60).all())
        self.assertEqual(list(sample.columns), ["id", "label", "email_text"])
        again, _ = prepare_dataset.build_sample(frame, per_class=10, seed=42, max_chars=60, min_chars=20)
        self.assertTrue(sample.equals(again))  # reproducible with the same seed

    def test_unknown_label_is_rejected(self):
        frame = pd.DataFrame({"Email Type": ["Spam"], "Email Text": ["x" * 40]})
        with self.assertRaises(ValueError):
            prepare_dataset.build_sample(frame, per_class=1, seed=1, max_chars=0, min_chars=1)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def make_config(self, tmp: Path) -> Path:
        dataset = tmp / "emails.csv"
        with dataset.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "label", "email_text"])
            writer.writerows(SAMPLE_ROWS)
        base = json.loads((ROOT / "configs" / "config.json").read_text(encoding="utf-8"))
        base.update({
            "model": "fake-model",
            "ollama_chat_url": f"http://127.0.0.1:{self.port}/api/chat",
            "dataset_path": str(dataset),
            "results_dir": str(tmp / "results"),
            "raw_output_path": str(tmp / "raw" / "run.jsonl"),
            "metadata_path": str(tmp / "raw" / "meta.json"),
            "warmup_requests": 2,
        })
        path = tmp / "config.json"
        path.write_text(json.dumps(base), encoding="utf-8")
        return path

    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config_path = self.make_config(tmp)
            predictions_path, summary_path = benchmark.run(config_path)

            raw_lines = [json.loads(line) for line in (tmp / "raw" / "run.jsonl").read_text().splitlines()]
            warmup = [r for r in raw_lines if r["phase"] == "warmup"]
            evaluation = [r for r in raw_lines if r["phase"] == "eval"]
            self.assertEqual(len(warmup), 4)  # 2 requests x 2 conditions
            self.assertEqual(len(evaluation), 16)  # 8 emails x 2 conditions
            for record in evaluation:
                self.assertGreater(record["latency_ms"], 0)
                self.assertIn("raw_response", record)

            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["conditions"]["basic"]["accuracy"], 1.0)
            self.assertEqual(summary["conditions"]["structured"]["phishing_detection_rate"], 0.75)
            with predictions_path.open(encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 16)

            metadata = json.loads((tmp / "raw" / "meta.json").read_text())
            self.assertEqual(metadata["model_digest"], "abc123")
            self.assertEqual(metadata["ollama_version"], "0.0.test")
            self.assertIn("finished_utc", metadata)

            out_dir = tmp / "processed"
            result = process_data.process(tmp / "raw" / "run.jsonl", out_dir)
            self.assertEqual(result["discarded_warmup"], 4)
            metrics = result["metrics"].set_index(["condition", "metric"])
            self.assertEqual(metrics.loc[("structured", "phishing_detection_rate"), "value"], 0.75)
            tests = result["tests"].set_index("metric")
            self.assertEqual(tests.loc["accuracy", "baseline_only"], 1)
            self.assertEqual(tests.loc["accuracy", "proposed_only"], 0)
            self.assertEqual(tests.loc["accuracy", "p_value"], 1.0)
            for name in ("metrics_summary.csv", "paired_tests.csv", "latency_by_length.csv", "results_table.md"):
                self.assertTrue((out_dir / name).exists(), name)

            saved = visualize.make_all(result, tmp / "figures")
            self.assertEqual(len(saved), 6)
            for path in saved:
                self.assertGreater(path.stat().st_size, 1000)

    def test_resume_skips_finished_emails(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config_path = self.make_config(tmp)
            benchmark.run(config_path)
            raw_path = tmp / "raw" / "run.jsonl"
            lines = raw_path.read_text().splitlines()
            keep = [line for line in lines if json.loads(line)["phase"] == "warmup"]
            evals = [line for line in lines if json.loads(line)["phase"] == "eval"]
            raw_path.write_text("\n".join(keep + evals[:6]) + "\n")  # simulate an interrupted run
            benchmark.run(config_path, resume=True)
            final = [json.loads(line) for line in raw_path.read_text().splitlines()]
            pairs = [(r["condition"], r["id"]) for r in final if r["phase"] == "eval"]
            self.assertEqual(len(pairs), 16)
            self.assertEqual(len(set(pairs)), 16)  # nothing duplicated


if __name__ == "__main__":
    unittest.main()
