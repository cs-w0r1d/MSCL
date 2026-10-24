"""Shared real-IHC PCA and signed cluster-level correspondence."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA


def shared_pca(predicted, real):
    predicted, real = np.asarray(predicted, dtype=float), np.asarray(real, dtype=float)
    if predicted.shape != real.shape or real.ndim != 3:
        raise ValueError("Expected matched [cases, clusters, embedding_dim] arrays")
    if real.shape[0] < 1 or real.shape[1] < 3 or real.shape[2] < 3:
        raise ValueError("Need at least one case, three paired clusters and three features")
    if not np.isfinite(real).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite cluster tokens")
    flat = real.reshape(-1, real.shape[-1])
    if not np.any(np.var(flat, axis=0) > 0):
        raise ValueError("Real-IHC tokens have no variance")
    pca = PCA(n_components=3, svd_solver="full").fit(flat)
    target = pca.transform(flat).reshape(*real.shape[:2], 3)
    prediction = pca.transform(predicted.reshape(flat.shape)).reshape(target.shape)
    limits = np.percentile(target.reshape(-1, 3), [1, 99], axis=0)
    span = limits[1] - limits[0]
    if np.any(span <= 0):
        raise ValueError("Degenerate PCA colour range")
    correlations = []
    for case in range(len(real)):
        values = []
        for pc in range(3):
            x, y = prediction[case, :, pc], target[case, :, pc]
            values.append(None if np.ptp(x) == 0 or np.ptp(y) == 0 else float(pearsonr(x, y).statistic))
        correlations.append({"case_index": case, "signed_r": values,
                             "mean_r": None if None in values else float(np.mean(values))})
    arrays = {"predicted_scores": prediction, "real_scores": target,
              "predicted_rgb": np.clip((prediction - limits[0]) / span, 0, 1),
              "real_rgb": np.clip((target - limits[0]) / span, 0, 1),
              "colour_limits": limits}
    metadata = {"cases": len(real), "clusters_per_case": real.shape[1],
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "correlations": correlations, "percentiles": [1, 99],
                "normalisation_reference": "pooled real-IHC cluster scores across all cases",
                "shared_scaling": True, "sign_flipping": False}
    return arrays, metadata
