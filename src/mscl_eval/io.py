"""Strict patient-level tables; never silently intersect model cohorts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_predictions(path, task):
    """Read long-format CSV and enforce identical patients/outcomes per model."""
    frame = pd.read_csv(path, dtype={"patient_id": str, "cohort": str, "model": str,
                                    "marker": str})
    if task not in {"survival", "markers"}:
        raise ValueError("task must be survival or markers")
    groups = ["cohort"] if task == "survival" else ["cohort", "marker"]
    outcomes = ["time", "event"] if task == "survival" else ["label"]
    score = "risk" if task == "survival" else "score"
    required = groups + ["patient_id", "model"] + outcomes + [score]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame = frame[required].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("Input is empty or contains missing values")
    for col in groups + ["patient_id", "model"]:
        if frame[col].str.strip().eq("").any():
            raise ValueError(f"Empty {col}")
    if frame.duplicated(groups + ["patient_id", "model"]).any():
        raise ValueError("Duplicate patient/model rows; aggregate slides before evaluation")
    for col in outcomes + [score]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[col]).all():
            raise ValueError(f"Non-finite {col}")
    binary = "event" if task == "survival" else "label"
    if not frame[binary].isin([0, 1]).all():
        raise ValueError(f"{binary} must be 0 or 1")
    if task == "survival" and not frame.time.gt(0).all():
        raise ValueError("Survival times must be positive; resolve eligibility upstream")
    if task == "markers" and not frame.score.between(0, 1).all():
        raise ValueError("Marker scores must be probabilities in [0, 1]")
    for _, part in frame.groupby(groups, sort=True):
        reference = None
        for _, rows in part.groupby("model", sort=True):
            current = rows.set_index("patient_id")[outcomes].sort_index()
            if reference is None:
                reference = current
            elif not current.equals(reference):
                raise ValueError("Models have different patient sets or outcomes within a cohort/task")
        if task == "markers" and reference.label.nunique() != 2:
            raise ValueError("Each marker/cohort must contain both outcome classes")
    return frame


def aligned(frame, score):
    models = sorted(frame.model.unique())
    first = frame[frame.model.eq(models[0])].sort_values("patient_id")
    risks = np.column_stack([
        frame[frame.model.eq(model)].set_index("patient_id").loc[first.patient_id, score]
        for model in models
    ])
    return first.reset_index(drop=True), models, risks


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_comparisons(path):
    if path is None:
        return []
    pairs = json.loads(Path(path).read_text())
    if not isinstance(pairs, list):
        raise ValueError("Comparisons must be a JSON list")
    seen = set()
    for item in pairs:
        if set(item) != {"baseline", "candidate"}:
            raise ValueError("Each comparison requires baseline and candidate")
        pair = (item["baseline"], item["candidate"])
        if not all(isinstance(x, str) and x for x in pair) or pair[0] == pair[1]:
            raise ValueError("Comparison models must be distinct non-empty strings")
        if pair in seen or pair[::-1] in seen:
            raise ValueError("Duplicate or reversed comparison")
        seen.add(pair)
    return pairs
