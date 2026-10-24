"""Command-line evaluation of previously generated predictions and tokens."""
import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .evaluation import evaluate_markers, evaluate_survival
from .io import file_hash, read_comparisons, read_predictions, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)
    for task in ("survival", "markers", "spatial"):
        command = sub.add_parser(task)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if task != "spatial":
            command.add_argument("--comparisons", type=Path)
            command.add_argument("--bootstrap", type=int, default=2000)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--no-plots", action="store_true")
        if task == "survival":
            command.add_argument("--horizons", type=float, nargs="+", default=[12, 24, 36])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; use a new directory for each analysis")
    metadata = {"package_version": "0.1.0", "task": args.task,
                "input_sha256": file_hash(args.input),
                "versions": {name: version(name) for name in
                             ["numpy", "pandas", "scipy", "scikit-learn", "lifelines"]}}
    if args.task == "spatial":
        from .spatial import shared_pca
        with np.load(args.input, allow_pickle=False) as pack:
            arrays, detail = shared_pca(pack["predicted"], pack["real"])
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez(args.output / "shared_pca.npz", **arrays)
        write_json(args.output / "spatial_summary.json", detail)
    else:
        if args.task == "survival" and (len(set(args.horizons)) != len(args.horizons) or
                                       any(not np.isfinite(h) or h <= 0 for h in args.horizons)):
            parser.error("Horizons must be distinct positive finite numbers")
        frame = read_predictions(args.input, args.task)
        comparisons = read_comparisons(args.comparisons)
        metadata.update(seed=args.seed, bootstrap=args.bootstrap, comparisons=comparisons,
                        inference="conditional on supplied fitted predictions; no model fitting",
                        comparison_test="two-sided centred bootstrap with plus-one correction",
                        multiplicity="BH across all supplied cohort/task/model comparisons")
        if args.task == "survival":
            summary, roc, paired = evaluate_survival(frame, comparisons, args.horizons,
                                                     args.bootstrap, args.seed)
            metadata.update(horizons_months=args.horizons,
                            roc="available-case; censoring at/before horizon excluded; not IPCW",
                            km="cohort/model median; risk >= median is high; unadjusted log-rank P")
            args.output.mkdir(parents=True, exist_ok=True)
            summary.to_csv(args.output / "survival_summary.csv", index=False)
            roc.to_csv(args.output / "roc_summary.csv", index=False)
            if not args.no_plots:
                from .plotting import plot_survival
                plot_survival(frame, summary, paired, args.horizons, args.output)
        else:
            summary, paired = evaluate_markers(frame, comparisons, args.bootstrap, args.seed)
            metadata.update(bootstrap_stratification="within observed binary label",
                            classification_threshold=0.5)
            args.output.mkdir(parents=True, exist_ok=True)
            summary.to_csv(args.output / "marker_summary.csv", index=False)
            if not args.no_plots:
                from .plotting import plot_markers
                plot_markers(frame, paired, args.output)
        paired.to_csv(args.output / "paired_comparisons.csv", index=False)
    write_json(args.output / "analysis_metadata.json", metadata)
    print(f"Saved {args.task} evaluation to {args.output}")


if __name__ == "__main__":
    main()
