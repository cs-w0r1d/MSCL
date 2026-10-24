"""Evaluation only: no classifier, Cox head, encoder or fusion model fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from .io import aligned
from .metrics import (available_case_roc, bh_adjust, centered_bootstrap_p,
                      cindex, km_summary, paired_bootstrap)


def comparison_rows(models, points, boot, comparisons, context):
    rows = []
    for pair in comparisons:
        if pair["baseline"] not in models or pair["candidate"] not in models:
            raise ValueError(f"Requested comparison not available in {context}: {pair}")
        a, b = models.index(pair["baseline"]), models.index(pair["candidate"])
        difference = boot[:, b] - boot[:, a]
        delta = points[b] - points[a]
        lo, hi = np.quantile(difference, [0.025, 0.975])
        rows.append({**context, **pair, "delta": delta, "ci_low": lo, "ci_high": hi,
                     "p_value": centered_bootstrap_p(difference, delta),
                     "valid_bootstraps": len(boot)})
    return rows


def adjust_comparisons(rows):
    columns = ["cohort", "baseline", "candidate", "delta", "ci_low", "ci_high",
               "p_value", "valid_bootstraps", "q_value"]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows)
    # One family across every explicitly requested comparison and cohort/task.
    result["q_value"] = bh_adjust(result.p_value)
    return result


def evaluate_survival(frame, comparisons=(), horizons=(12, 24, 36), n_boot=2000, seed=42):
    summary, roc_rows, paired = [], [], []
    for cohort, part in frame.groupby("cohort", sort=True):
        rows, models, risks = aligned(part, "risk")
        time, event = rows.time.to_numpy(float), rows.event.to_numpy(int)
        points = [cindex(time, event, r) for r in risks.T]
        boot = paired_bootstrap(time, event, risks, n_boot, seed)
        for index, model in enumerate(models):
            lo, hi = np.quantile(boot[:, index], [0.025, 0.975])
            summary.append({"cohort": cohort, "model": model, "n": len(rows),
                            "events": int(event.sum()), "c_index": points[index],
                            "ci_low": lo, "ci_high": hi, "valid_bootstraps": len(boot),
                            **km_summary(time, event, risks[:, index])})
            for horizon in horizons:
                roc = available_case_roc(time, event, risks[:, index], horizon)
                roc_rows.append({"cohort": cohort, "model": model, "horizon": horizon,
                                 **{k: v for k, v in roc.items() if k not in ("fpr", "tpr")}})
        paired.extend(comparison_rows(models, points, boot, comparisons, {"cohort": cohort}))
    return pd.DataFrame(summary), pd.DataFrame(roc_rows), adjust_comparisons(paired)


def evaluate_markers(frame, comparisons=(), n_boot=2000, seed=42):
    if n_boot < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    summary, paired = [], []
    for (cohort, marker), part in frame.groupby(["cohort", "marker"], sort=True):
        rows, models, scores = aligned(part, "score")
        label = rows.label.to_numpy(int)
        rng = np.random.default_rng(seed)
        groups = [np.flatnonzero(label == value) for value in (0, 1)]
        boot = []
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            boot.append([roc_auc_score(label[idx], s[idx]) for s in scores.T])
        boot = np.asarray(boot)
        points = [float(roc_auc_score(label, s)) for s in scores.T]
        for index, model in enumerate(models):
            lo, hi = np.quantile(boot[:, index], [0.025, 0.975])
            summary.append({"cohort": cohort, "marker": marker, "model": model,
                            "n": len(label), "positive": int(label.sum()),
                            "auroc": points[index], "ci_low": lo, "ci_high": hi,
                            "average_precision": average_precision_score(label, scores[:, index]),
                            "balanced_accuracy": balanced_accuracy_score(label, scores[:, index] >= 0.5)})
        paired.extend(comparison_rows(models, points, boot, comparisons,
                                      {"cohort": cohort, "marker": marker}))
    return pd.DataFrame(summary), adjust_comparisons(paired)
