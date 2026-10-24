"""Statistics conditional on supplied, already fitted model predictions."""
from __future__ import annotations

import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, roc_curve


def cindex(time, event, risk):
    # Higher supplied scores must mean greater hazard, not longer survival.
    return float(concordance_index(time, -np.asarray(risk), event))


def available_case_roc(time, event, risk, horizon):
    """Descriptive binary ROC, NOT IPCW time-dependent ROC."""
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("Horizon must be positive and finite")
    time, event, risk = map(np.asarray, (time, event, risk))
    cases = (event == 1) & (time <= horizon)
    controls = time > horizon
    keep = cases | controls
    result = {"n": len(time), "cases": int(cases.sum()),
              "controls": int(controls.sum()), "excluded": int((~keep).sum()),
              "auc": None, "fpr": None, "tpr": None}
    if cases.any() and controls.any():
        label = cases[keep].astype(int)
        fpr, tpr, _ = roc_curve(label, risk[keep])
        result.update(auc=float(roc_auc_score(label, risk[keep])), fpr=fpr, tpr=tpr)
    return result


def km_summary(time, event, risk):
    threshold = float(np.median(risk))
    high = np.asarray(risk) >= threshold
    p = None
    if high.any() and (~high).any() and np.sum(event) > 0:
        p = float(logrank_test(np.asarray(time)[high], np.asarray(time)[~high],
                              np.asarray(event)[high], np.asarray(event)[~high]).p_value)
        if not np.isfinite(p):
            p = None
    return {"threshold": threshold, "n_high": int(high.sum()),
            "n_low": int((~high).sum()), "logrank_p": p}


def paired_bootstrap(time, event, risks, n_boot=2000, seed=42):
    if n_boot < 100:
        raise ValueError("Use at least 100 resamples (2000 recommended)")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(time), len(time))
        try:
            samples.append([cindex(time[indices], event[indices], r[indices])
                            for r in risks.T])
        except ZeroDivisionError:
            continue  # No comparable pairs; the same resample is removed for every model.
    if len(samples) < 0.8 * n_boot:
        raise ValueError("Fewer than 80% of bootstrap resamples contain comparable pairs")
    return np.asarray(samples)


def centered_bootstrap_p(delta_samples, observed_delta):
    """Two-sided centred-bootstrap test, with a plus-one correction."""
    deviations = np.abs(np.asarray(delta_samples) - observed_delta)
    return float((np.count_nonzero(deviations >= abs(observed_delta)) + 1)
                 / (len(deviations) + 1))


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not len(p):
        return p
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("P values must be finite probabilities")
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    result = np.empty_like(p)
    result[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return result
