from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

import numpy as np

try:
    import torch
    from mscl_eval.inference import InferenceConfig, MSCLInference, load_checkpoint, representative_indices
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch inference extra not installed")
class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = InferenceConfig(input_dim=8, embed_dim=8, local_layers=1,
                                   global_layers=1, heads=2, num_clusters=3,
                                   patches_per_cluster=2, dropout=0.1)

    def test_shape_and_deterministic_inference(self):
        model = MSCLInference(self.cfg)
        x = torch.randn(2, 3, 2, 8)
        a, b = model(x), model(x)
        self.assertEqual(a["he"].shape, (2, 8))
        self.assertEqual(a["pred_ihc_clusters"].shape, (2, 3, 8))
        torch.testing.assert_close(a["he"], b["he"], rtol=0, atol=0)
        self.assertFalse(a["he"].requires_grad)

    def test_wrong_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected"):
            MSCLInference(self.cfg)(torch.randn(1, 2, 3, 8))

    def test_checkpoint_roundtrip(self):
        model = MSCLInference(self.cfg)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.pt"
            torch.save({"model_config": asdict(self.cfg), "model": model.state_dict()}, path)
            loaded = load_checkpoint(path)
            x = torch.randn(1, 3, 2, 8)
            torch.testing.assert_close(model(x)["pred_ihc"], loaded(x)["pred_ihc"])

    def test_incompatible_checkpoint_rejected(self):
        state = MSCLInference(self.cfg).state_dict()
        state.pop(next(iter(state)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.pt"
            torch.save({"model_config": asdict(self.cfg), "model": state}, path)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                load_checkpoint(path)

    def test_extraction_command_on_synthetic_features(self):
        from mscl_eval.inference import main
        model = MSCLInference(self.cfg)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "synthetic.pt"
            torch.save({"model_config": asdict(self.cfg), "model": model.state_dict()}, checkpoint)
            features = np.random.default_rng(42).normal(size=(100, 8)).astype(np.float32)
            np.save(root / "features.npy", features)
            manifest = root / "slides.csv"
            manifest.write_text("slide_id,features_path\nSYNTHETIC_SLIDE,features.npy\n")
            output = root / "output"
            argv = ["mscl-extract", "--checkpoint", str(checkpoint), "--manifest", str(manifest),
                    "--output", str(output)]
            with patch("sys.argv", argv):
                main()
            with np.load(output / "SYNTHETIC_SLIDE.npz", allow_pickle=False) as pack:
                self.assertEqual(set(pack.files), {"he", "pred_ihc"})
                self.assertEqual(pack["he"].shape, (8,))
            self.assertEqual(json.loads((output / "protocol.json").read_text())["slides"], 1)

    def test_sample_small_clusters_by_repetition(self):
        features = np.array([[0., 0.], [10., 10.], [20., 20.]], dtype=np.float32)
        # Isolate repetition from stochastic mini-batch clustering of only three rows.
        with patch("mscl_eval.inference.MiniBatchKMeans") as clusterer:
            clusterer.return_value.fit_predict.return_value = np.arange(3)
            clusterer.return_value.cluster_centers_ = features.copy()
            indices = representative_indices(features, 3, 4)
        self.assertEqual(indices.shape, (3, 4))
        for row in indices:
            self.assertEqual(len(np.unique(row)), 1)

    def test_empty_cluster_rejected(self):
        with patch("mscl_eval.inference.MiniBatchKMeans") as clusterer:
            clusterer.return_value.fit_predict.return_value = np.zeros(3, dtype=int)
            clusterer.return_value.cluster_centers_ = np.zeros((3, 8))
            with self.assertRaisesRegex(ValueError, "Empty morphology"):
                representative_indices(np.zeros((3, 8)), 3, 2)

    def test_clustering_is_reproducible(self):
        features = np.random.default_rng(42).normal(size=(100, 8))
        a = representative_indices(features, 3, 2)
        b = representative_indices(features, 3, 2)
        np.testing.assert_array_equal(a, b)

    def test_sample_too_few_patches_rejected(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            representative_indices(np.zeros((2, 8)), 3, 2)


if __name__ == "__main__":
    unittest.main()
