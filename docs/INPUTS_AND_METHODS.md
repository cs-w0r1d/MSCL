# Input formats and statistical definitions

## Survival

Required CSV columns: `cohort,patient_id,model,time,event,risk`.

- One row per patient and model, with positive finite time in months.
- `event` is exactly 0 (censored) or 1 (death).
- Risk is finite, with higher values indicating greater hazard. Do not pass
  survival probabilities without converting their direction appropriately.
- IDs are opaque strings. Patient sets and outcomes must agree exactly across
  models within each cohort. Models may differ between cohorts unless an
  explicit comparison requires them in all cohorts.
- The tool does not decide eligibility, impute outcomes, average slides,
  optimise thresholds, select model versions, or fit a risk head.

Harrell's C-index is computed using lifelines with the risk sign reversed
for its survival-score convention. Percentile confidence intervals use paired
patient bootstrap resampling. A resample without comparable survival pairs
is discarded for all models together; valid resample counts are reported.
The analysis fails if fewer than 80% of requested resamples are usable.

For each ROC horizon `h`, cases are observed deaths with `time <= h` and
controls have `time > h`. Censored patients at or before `h` are excluded.
Deaths exactly at `h` are cases; censoring exactly at `h` is excluded.
Cases, controls and exclusions are saved in `roc_summary.csv`. If one class
is absent, AUC is unavailable and no curve is drawn. This is a descriptive
available-case estimator and may be biased by censoring; it is **not** IPCW.

KM curves use `risk >= cohort/model median` as the high-risk group. All risk
ties stay together; groups need not have equal sizes. A single nonempty group
has no log-rank P value. Bands are lifelines' 95% confidence intervals, not
decorative shading. Log-rank P values are unadjusted and are labelled as such
in the analysis metadata. Fix an external threshold in a separate, prespecified
analysis if clinically transferable thresholds are required.

## Paired comparisons, including foundation-feature fusion

Pass a JSON list such as:

```json
[
  {"baseline": "GigaPath", "candidate": "GigaPath_plus_MSCL"},
  {"baseline": "PRISM", "candidate": "PRISM_plus_MSCL"}
]
```

Each listed comparison must exist in every supplied cohort/task. The metric
difference is candidate minus baseline, on matched patients and the same
bootstrap indices. The two-sided centred-bootstrap P value is:

`(1 + count(abs(delta_bootstrap - delta_observed) >= abs(delta_observed))) / (B + 1)`.

Percentile intervals describe the bootstrap difference distribution.
Benjamini–Hochberg correction is applied once across every requested comparison
and cohort (and marker, in marker evaluation). Both P and q are saved; figure
annotations are calculated, never hard-coded. This estimator is explicitly
defined and may differ from legacy analysis scripts' bootstrap-tail tests.
No comparisons are inferred from model names when the JSON file is omitted.

## Marker evaluation

Required columns: `cohort,marker,patient_id,model,label,score`.

`label` is 0 or 1. `score` is a probability in [0, 1] for label 1. Both classes
must be present. A patient's label must agree across models. This tool reports
AUROC with stratified patient-bootstrap intervals, average precision, and
balanced accuracy at a fixed probability threshold of 0.5. AUROC comparisons
use the same explicit comparison mechanism as survival evaluation.

The input must contain already curated marker-specific labels. Missing,
equivocal or unquantified pathology-report values must not be silently treated
as negatives. The public package neither parses hospital reports nor selects
HER2/Ki-67 thresholds. Record label definitions and adjudication separately.

## Spatial correspondence

Supply a NumPy `.npz` file with two numeric arrays, `predicted` and `real`,
both `[cases, clusters, embedding_dim]`. Entries must be paired in exactly
the same order and represent the same latent target space, for example the
IHC local encoder's cluster-token space. A predicted local IHC token must not
be compared directly with an unrelated frozen patch encoder's output.

A three-component PCA is fitted once to all real-IHC cluster tokens. The same
basis projects both arrays; there is no per-case sign flip. Pearson correlations
are **signed** and computed across paired clusters within each case. The mean
is the arithmetic mean of the three signed correlations, not of absolute values.
Constant projected inputs yield an unavailable correlation.

RGB lower/upper bounds are the 1st/99th percentiles of pooled real-IHC scores
for each component. The same bounds are applied to every case and both sides,
then values are clipped to [0, 1]. No per-case or prediction-specific rescaling
is performed. Cluster-level RGB values can be mapped to member patch coordinates
upstream; the number of rendered patches is not the sample size for correlation.

## Provenance

Every evaluation saves the input SHA-256, software versions, seed, resample
count and explicit statistical conventions. This does not substitute for a
curated cohort manifest or model provenance. Output tables are aggregate
summaries; input participant-level files are not copied into the output folder.
