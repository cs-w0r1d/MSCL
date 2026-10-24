import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from mscl_eval.evaluation import evaluate_markers, evaluate_survival
from mscl_eval.io import read_comparisons, read_predictions
from mscl_eval.metrics import (available_case_roc, bh_adjust, centered_bootstrap_p,
                               cindex, km_summary)
from mscl_eval.spatial import shared_pca


class InputTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame([dict(cohort="Synthetic", patient_id=f"SYNTHETIC_{i}",
                                  model=model, time=i + 1, event=i % 2, risk=-i)
                             for model in ("A", "B") for i in range(8)])

    def read(self, frame):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            frame.to_csv(path, index=False)
            return read_predictions(path, "survival")

    def test_valid_and_independent_row_order(self):
        frame = self.table().sample(frac=1, random_state=1)
        self.assertEqual(len(self.read(frame)), 16)

    def test_mismatched_patients_rejected(self):
        with self.assertRaisesRegex(ValueError, "different patient"):
            self.read(self.table().iloc[1:])

    def test_mismatched_outcomes_rejected(self):
        frame = self.table()
        frame.loc[0, "time"] = 99
        with self.assertRaisesRegex(ValueError, "different patient"):
            self.read(frame)

    def test_duplicate_rows_rejected(self):
        frame = self.table()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.read(pd.concat([frame, frame.iloc[:1]]))

    def test_fractional_events_rejected(self):
        frame = self.table().astype({"event": float})
        frame.loc[0, "event"] = 0.5
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            self.read(frame)

    def test_nonpositive_time_rejected(self):
        frame = self.table()
        frame.loc[0, "time"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            self.read(frame)

    def test_infinite_risk_rejected(self):
        frame = self.table().astype({"risk": float})
        frame.loc[0, "risk"] = np.inf
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.read(frame)

    def test_reversed_comparisons_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparisons.json"
            path.write_text(json.dumps([dict(baseline="A", candidate="B"),
                                        dict(baseline="B", candidate="A")]))
            with self.assertRaisesRegex(ValueError, "reversed"):
                read_comparisons(path)


class StatisticsTests(unittest.TestCase):
    def test_cindex_direction(self):
        self.assertEqual(cindex([1, 2, 3], [1, 1, 1], [3, 2, 1]), 1)
        self.assertEqual(cindex([1, 2, 3], [1, 1, 1], [1, 2, 3]), 0)

    def test_censoring_and_horizon_boundary(self):
        result = available_case_roc([6, 12, 12, 24, 5], [1, 1, 0, 0, 0],
                                    [0.9, 0.8, 0.7, 0.1, 0.6], 12)
        self.assertEqual((result["cases"], result["controls"], result["excluded"]), (2, 1, 2))
        self.assertEqual(result["auc"], 1)

    def test_roc_missing_class(self):
        result = available_case_roc([2, 3], [0, 0], [0.2, 0.1], 12)
        self.assertIsNone(result["auc"])

    def test_risk_ties_not_forced_into_two_groups(self):
        result = km_summary([1, 2, 3], [1, 1, 0], [1, 1, 1])
        self.assertIsNone(result["logrank_p"])
        self.assertEqual(result["n_low"], 0)

    def test_bh(self):
        np.testing.assert_allclose(bh_adjust([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])

    def test_centered_p_plus_one_and_null(self):
        self.assertEqual(centered_bootstrap_p(np.zeros(100), 0), 1)
        self.assertGreater(centered_bootstrap_p(np.ones(100), 1), 0)

    def test_identical_models_zero_difference(self):
        rows = [dict(cohort="Synthetic", patient_id=f"SYNTHETIC_{i}", model=model,
                     time=i + 1, event=1, risk=-i)
                for model in ("A", "B") for i in range(12)]
        summary, roc, paired = evaluate_survival(pd.DataFrame(rows),
                    [dict(baseline="A", candidate="B")], [5], 100, 42)
        self.assertEqual(len(summary), 2)
        self.assertEqual(len(roc), 2)
        self.assertEqual(paired.iloc[0].delta, 0)
        self.assertEqual(paired.iloc[0].q_value, 1)

    def test_marker_stratified_bootstrap(self):
        rows = [dict(cohort="Synthetic", marker="Synthetic_marker", patient_id=f"SYNTHETIC_{i}",
                     model=model, label=i % 2, score=0.9 if i % 2 else 0.1)
                for model in ("A", "B") for i in range(12)]
        summary, paired = evaluate_markers(pd.DataFrame(rows),
                                          [dict(baseline="A", candidate="B")], 100)
        np.testing.assert_allclose(summary.auroc, 1)
        self.assertEqual(paired.iloc[0].q_value, 1)

    def test_missing_comparison_not_silently_dropped(self):
        frame = InputTests().table()
        with self.assertRaisesRegex(ValueError, "not available"):
            evaluate_survival(frame, [dict(baseline="A", candidate="Unknown")], [2], 100)


class SpatialTests(unittest.TestCase):
    def test_signed_negative_correlations_retained(self):
        real = np.random.default_rng(42).normal(size=(2, 20, 8))
        _, detail = shared_pca(-real, real)
        np.testing.assert_allclose(detail["correlations"][0]["signed_r"], [-1, -1, -1])

    def test_common_scaling(self):
        real = np.random.default_rng(42).normal(size=(2, 20, 8))
        arrays, detail = shared_pca(real, real)
        np.testing.assert_allclose(arrays["predicted_rgb"], arrays["real_rgb"])
        np.testing.assert_allclose(arrays["colour_limits"],
                                   np.percentile(arrays["real_scores"].reshape(-1, 3), [1, 99], axis=0))
        self.assertTrue(detail["shared_scaling"])

    def test_mismatched_tokens_rejected(self):
        with self.assertRaisesRegex(ValueError, "matched"):
            shared_pca(np.ones((1, 10, 5)), np.ones((2, 10, 5)))


if __name__ == "__main__":
    unittest.main()
