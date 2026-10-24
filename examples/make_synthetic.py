"""Generate fictional data for testing; never manuscript or patient results."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    survival, markers = [], []
    models = ["Baseline", "MSCL_HE", "Baseline_plus_MSCL"]
    for cohort in ("Synthetic_A", "Synthetic_B"):
        latent = rng.normal(size=80)
        event_time = rng.exponential(35 * np.exp(-latent)) + 0.1
        censor_time = rng.uniform(5, 65, 80)
        time = np.minimum(event_time, censor_time)
        event = (event_time <= censor_time).astype(int)
        risks = [latent + rng.normal(size=80) for _ in models]
        for i in range(80):
            for model, risk in zip(models, risks):
                survival.append(dict(cohort=cohort, patient_id=f"SYNTHETIC_{i:03}",
                                     model=model, time=time[i], event=event[i], risk=risk[i]))
        for marker in ("Synthetic_marker_A", "Synthetic_marker_B"):
            label = (latent + rng.normal(size=80) > 0).astype(int)
            scores = [1 / (1 + np.exp(-(latent + rng.normal(size=80)))) for _ in models]
            for i in range(80):
                for model, score in zip(models, scores):
                    markers.append(dict(cohort=cohort, marker=marker, patient_id=f"SYNTHETIC_{i:03}",
                                        model=model, label=label[i], score=score[i]))
    pd.DataFrame(survival).to_csv(args.output / "survival.csv", index=False)
    pd.DataFrame(markers).to_csv(args.output / "markers.csv", index=False)
    real = rng.normal(size=(4, 50, 16))
    predicted = real + rng.normal(scale=0.5, size=real.shape)
    np.savez(args.output / "cluster_tokens.npz", predicted=predicted, real=real)
    print(f"Fictional demonstration data written to {args.output}")


if __name__ == "__main__":
    main()
