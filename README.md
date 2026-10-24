# MSCL: Multi-Stain Contrastive Learning

Public **inference and evaluation** utilities for IHC-informed computational
pathology representations. MSCL uses paired H&E–IHC supervision to learn
disease-focused representations and supports H&E-derived slide embeddings
and predicted-IHC latent features.

This initial release contains selected, portable analysis code. **Training
code and pretrained weights are not included.** The evaluation tools work
with your own saved predictions; feature extraction additionally requires
a compatible, separately obtained hierarchical MSCL checkpoint.

## Included

- Inference-only hierarchical aggregation of precomputed CONCH patch features:
  patch features → mean-initialised local cluster tokens → global WSI token.
- H&E, optional real-IHC, and H&E-derived predicted-IHC slide representations.
- Patient-level Harrell's C-index, Kaplan–Meier curves, log-rank tests, descriptive
  available-case ROC, paired bootstrap comparisons and BH correction.
- Binary marker-prediction evaluation from existing probability scores.
- Shared-basis PCA, signed cluster-level correlations, and common RGB scaling
  for already matched predicted/real IHC tokens.
- Synthetic examples, input validation, unit tests, and PDF/PNG/SVG plotting.

Not included: pretraining, masking/reconstruction losses, training launchers,
optimisers, downstream risk-head or classifier fitting, trained weights,
clinical data, patient identifiers, pathology reports, WSIs, private feature
caches, or participant-level predictions. See [release scope](docs/RELEASE_SCOPE.md).

## Installation

Python 3.10 or newer is required. A virtual environment is recommended.

```bash
git clone https://github.com/cs-w0r1d/MSCL.git
cd MSCL
python -m pip install -e .
```

For feature extraction, also install PyTorch for your CPU/CUDA environment
and the inference extra:

```bash
python -m pip install -e '.[inference]'
```

The statistical tools do not require PyTorch, a GPU, model weights, or
access to the original hospital datasets.

## Quick start with synthetic data

Everything produced by this example is fictional and is **not a paper result**.

```bash
python examples/make_synthetic.py --output outputs/synthetic

mscl-evaluate survival \
  --input outputs/synthetic/survival.csv \
  --comparisons examples/comparisons.json \
  --bootstrap 2000 --horizons 12 24 36 \
  --output outputs/survival_demo

mscl-evaluate markers \
  --input outputs/synthetic/markers.csv \
  --comparisons examples/comparisons.json \
  --bootstrap 2000 --output outputs/marker_demo

mscl-evaluate spatial \
  --input outputs/synthetic/cluster_tokens.npz \
  --output outputs/spatial_demo
```

Use a new or empty output directory for each run. Survival/marker commands
save summary CSVs, a comparison table, analysis metadata and plots. `--no-plots`
skips rendering. The spatial command saves PCA scores, RGB arrays and signed
correlations, not a reconstructed IHC image.

## Use your own predictions

Survival input is a long-format CSV with one row per patient/model:

```text
cohort,patient_id,model,time,event,risk
Demo,SYNTHETIC_001,Model_A,24,1,0.8
Demo,SYNTHETIC_001,Model_B,24,1,0.6
```

Time is in months, `event=1` means death, and **larger risk scores mean greater
hazard**. Average multiple slides within patient upstream. Models within a
cohort must have identical patient sets and outcomes; mismatches are errors,
not silently resolved by taking an intersection.

To compare fusion against a foundation representation, provide risks from
your already fitted, matched foundation-only and fused models and list the
comparisons explicitly in JSON. This release **does not fit a fusion model**.

See [input formats and statistical definitions](docs/INPUTS_AND_METHODS.md)
for marker inputs, analysis denominators, bootstrap tests and plotting rules.

## WSI feature extraction

Patch feature extraction with CONCH is upstream of this package. Obtain and
use that model under its own licence and access terms; it is not redistributed
here. Input `.npy` arrays or HDF5 `features` datasets have shape `[patches, 512]`
for the CONCH-v1 hierarchical configuration. Feature dimensions alone do not
establish compatibility: the checkpoint and patch encoder must match.

Create a manifest with `slide_id,features_path` and an optional
`ihc_features_path` column. Relative paths are resolved against the manifest
directory. The template in `examples/` illustrates the schema but does not
ship feature files.

```bash
mscl-extract \
  --manifest data/slides.csv \
  --checkpoint checkpoints/hierarchical_mscl.pt \
  --device cpu \
  --output outputs/slide_features
```

The default configuration selects 50 morphology clusters and 10 nearest
representative patches per cluster. Small clusters repeat their representatives;
empty clusters fail explicitly. Each output `.npz` contains `he`, `pred_ihc`,
and, when supplied, `ihc`. H&E inference does not use real-IHC input or
coordinates. Optional real-IHC features are sampled independently for
slide-level fusion; those independently sampled clusters are **not** matched
spatial targets.

Only the hierarchical checkpoint format is supported. Legacy single-level
checkpoints are not interchangeable. Loading uses `torch.load(weights_only=True)`
and strict inference-module matching; no unsafe pickle fallback is provided.
No pretrained checkpoint download is currently provided by this release.

## Interpretation and reproducibility

- This is an analysis toolkit, not a claim that a single public command
  reproduces every historical manuscript panel. Exact reproduction also
  needs the applicable cohort manifest, feature/checkpoint version, fitted
  heads, preprocessing, predictions and analysis configuration.
- ROC here is **descriptive available-case ROC**, not censoring-adjusted
  time-dependent ROC. Early-censored observations are excluded consistently
  in every cohort. Do not interpret these estimates as IPCW estimates.
- Comparisons use one explicitly supplied family across cohorts/tasks. Do not
  choose patients, horizons, cutoffs, models or comparison families by the
  significance of the resulting evaluation.
- Bootstrap uncertainty is conditional on the supplied fitted models. KM
  separation is not a paired test of superiority between prediction models.
- Predicted-IHC features and PCA colours are latent representations, not
  calibrated protein expression, virtual staining, or clinical diagnoses.

## Tests and release checks

```bash
python -m unittest discover -s tests -v
python tools/check_release.py
```

Core statistical tests run without PyTorch; inference tests run when PyTorch
is installed. CI uses synthetic data only. Generated outputs, checkpoints,
feature files and private data are excluded from Git by default.

## Licence and citation

Source code in this repository is released under the [MIT licence](LICENSE).
Third-party packages, foundation models, weights and datasets retain their
own licences. Cite the final associated manuscript when bibliographic details
become available and record the repository commit used for your analysis.

Research use only; this software has not been validated for clinical decisions.
