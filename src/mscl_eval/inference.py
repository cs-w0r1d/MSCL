"""Inference-only hierarchical MSCL modules; no losses, decoders or fitting."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.cluster import MiniBatchKMeans

from .io import file_hash, write_json


@dataclass(frozen=True)
class InferenceConfig:
    input_dim: int = 512
    embed_dim: int = 512
    local_layers: int = 8
    global_layers: int = 4
    heads: int = 8
    num_clusters: int = 50
    patches_per_cluster: int = 10
    dropout: float = 0.1


def transformer(width, layers, heads, dropout):
    block = nn.TransformerEncoderLayer(width, heads, dim_feedforward=4 * width,
                                      dropout=dropout, activation="gelu",
                                      batch_first=True, norm_first=True)
    return nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(width))


class StainEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_projection = nn.Linear(cfg.input_dim, cfg.embed_dim)
        self.local_cls_position = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.local_patch_position = nn.Parameter(torch.zeros(1, cfg.patches_per_cluster, cfg.embed_dim))
        self.local_encoder = transformer(cfg.embed_dim, cfg.local_layers, cfg.heads, cfg.dropout)
        self.wsi_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.global_position = nn.Parameter(torch.zeros(1, cfg.num_clusters + 1, cfg.embed_dim))
        self.global_encoder = transformer(cfg.embed_dim, cfg.global_layers, cfg.heads, cfg.dropout)
        self.wsi_norm = nn.LayerNorm(cfg.embed_dim)

    def aggregate(self, clusters):
        wsi = self.wsi_token.expand(len(clusters), -1, -1)
        tokens = torch.cat([wsi, clusters], dim=1) + self.global_position
        return self.wsi_norm(self.global_encoder(tokens)[:, 0])

    def forward(self, patches):
        cfg = self.cfg
        expected = (cfg.num_clusters, cfg.patches_per_cluster, cfg.input_dim)
        if patches.ndim != 4 or tuple(patches.shape[1:]) != expected:
            raise ValueError(f"Expected [batch, {expected}], received {tuple(patches.shape)}")
        if not torch.isfinite(patches).all():
            raise ValueError("Patch embeddings contain non-finite values")
        flat = patches.reshape(-1, cfg.patches_per_cluster, cfg.input_dim)
        seed = self.input_projection(flat.mean(1, keepdim=True)) + self.local_cls_position
        projected = self.input_projection(flat) + self.local_patch_position
        local = self.local_encoder(torch.cat([seed, projected], dim=1))[:, 0]
        local = local.reshape(len(patches), cfg.num_clusters, cfg.embed_dim)
        return self.aggregate(local), local


class CrossStainPredictor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width),
                                 nn.Linear(width, width))

    def forward(self, clusters):
        return self.net(clusters)


class MSCLInference(nn.Module):
    """Only the modules needed for H&E, real-IHC and predicted-IHC inference."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.he_encoder = StainEncoder(cfg)
        self.ihc_encoder = StainEncoder(cfg)
        self.cross_predictor = CrossStainPredictor(cfg.embed_dim)
        self.eval()

    @torch.inference_mode()
    def forward(self, he, ihc=None):
        if self.training:
            raise RuntimeError("This public module is inference-only; call eval()")
        he_wsi, he_local = self.he_encoder(he)
        pred_local = self.cross_predictor(he_local)
        output = {"he": he_wsi, "pred_ihc": self.ihc_encoder.aggregate(pred_local),
                  "he_clusters": he_local, "pred_ihc_clusters": pred_local}
        if ihc is not None:
            output["ihc"], output["ihc_clusters"] = self.ihc_encoder(ihc)
        return output


def load_checkpoint(path, device="cpu"):
    """Load a tensor/dict checkpoint safely; never fall back to arbitrary pickle."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or not {"model_config", "model"} <= checkpoint.keys():
        raise ValueError("Expected a checkpoint with model_config and model keys")
    config = checkpoint["model_config"]
    names = {f.name for f in fields(InferenceConfig)}
    if not names <= config.keys():
        raise ValueError(f"Missing inference configuration keys: {sorted(names - config.keys())}")
    model = MSCLInference(InferenceConfig(**{k: config[k] for k in names}))
    state = checkpoint["model"]
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = [k for k in state if k not in expected and not
             (k.startswith(("he_decoder.", "ihc_decoder.")) or k == "logit_scale")]
    if missing or extra:
        raise ValueError(f"Incompatible hierarchical checkpoint; missing={missing}, unexpected={extra}")
    model.load_state_dict({k: state[k] for k in expected}, strict=True)
    model.requires_grad_(False)
    return model.eval().to(device)


def representative_indices(features, n_clusters=50, patches_per_cluster=10, seed=42):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Features must be a finite two-dimensional matrix")
    if n_clusters < 1 or patches_per_cluster < 1 or len(features) < n_clusters:
        raise ValueError("Insufficient patches or invalid cluster dimensions")
    clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                               batch_size=min(4096, len(features)), n_init=3,
                               max_iter=120, reassignment_ratio=0.01)
    labels = clusterer.fit_predict(features)
    chosen = np.empty((n_clusters, patches_per_cluster), dtype=np.int64)
    for cluster in range(n_clusters):
        indices = np.flatnonzero(labels == cluster)
        if not len(indices):
            raise ValueError(f"Empty morphology cluster {cluster}; inspect input features")
        distances = ((features[indices] - clusterer.cluster_centers_[cluster]) ** 2).sum(1)
        nearest = indices[np.argsort(distances)]
        chosen[cluster] = np.resize(nearest, patches_per_cluster)
    return chosen


def load_features(path, input_dim):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        features = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            features = np.asarray(handle["features"])
    else:
        raise ValueError("Patch features must be .npy or HDF5 with a features dataset")
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != input_dim or not np.isfinite(features).all():
        raise ValueError(f"Expected finite [patches, {input_dim}] input: {path.name}")
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = load_checkpoint(args.checkpoint, args.device)
    manifest = pd.read_csv(args.manifest, dtype=str)
    if not {"slide_id", "features_path"} <= set(manifest):
        parser.error("Manifest needs slide_id and features_path columns")
    if manifest.empty or manifest[["slide_id", "features_path"]].isna().any().any():
        parser.error("Manifest is empty or has missing required entries")
    if manifest.slide_id.duplicated().any() or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", x) for x in manifest.slide_id
    ):
        parser.error("slide_id must be unique and safe for use as a filename")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg

    def tensor(value):
        path = Path(value)
        if not path.is_absolute():
            path = args.manifest.parent / path
        features = load_features(path, cfg.input_dim)
        indices = representative_indices(features, cfg.num_clusters, cfg.patches_per_cluster, args.seed)
        return torch.from_numpy(features[indices])[None].to(args.device), len(features)

    for row in manifest.to_dict("records"):
        he, count = tensor(row["features_path"])
        ihc = None
        if pd.notna(row.get("ihc_features_path")) and row.get("ihc_features_path"):
            ihc, _ = tensor(row["ihc_features_path"])
        result = model(he, ihc)
        # Independently sampled real-IHC clusters are not spatially aligned targets.
        vectors = {key: value[0].cpu().numpy() for key, value in result.items()
                   if not key.endswith("clusters")}
        np.savez(args.output / (row["slide_id"] + ".npz"), **vectors)
        print(f"Encoded {row['slide_id']}: {count} H&E patches")
    write_json(args.output / "protocol.json", {
        "package_version": "0.1.0", "checkpoint_sha256": file_hash(args.checkpoint),
        "manifest_sha256": file_hash(args.manifest), "slides": len(manifest),
        "seed": args.seed, "device": args.device,
        "sampling": "independent per stain; nearest cluster representatives; repeat small clusters",
        "pretraining_included": False,
    })


if __name__ == "__main__":
    main()
