import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark import calculate_metrics, parse_label


class BenchmarkTests(unittest.TestCase):
    def test_json_label_parsing(self):
        self.assertEqual(parse_label('{"label": "phishing"}'), "phishing")
        self.assertEqual(parse_label('{"label": "legitimate"}'), "legitimate")

    def test_metrics(self):
        rows = [
            {"true_label": "phishing", "predicted_label": "phishing"},
            {"true_label": "phishing", "predicted_label": "legitimate"},
            {"true_label": "legitimate", "predicted_label": "legitimate"},
            {"true_label": "legitimate", "predicted_label": "phishing"},
        ]
        result = calculate_metrics(rows)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["phishing_detection_rate"], 0.5)
        self.assertEqual(result["false_positive_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
