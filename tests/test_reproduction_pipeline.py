import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from classification.evaluate_augmented import build_augmented_train, compute_metrics, generated_path_for_ratio
from data.prepare_data import stratified_split
from diffusion.diffusion_process import SpectralDiffusion
from diffusion.generate_spectra import generated_count_by_class
from diffusion.model import SpectralDiffusionUNet


def test_stratified_split_has_no_index_leakage() -> None:
    df = pd.DataFrame(
        {
            "label": ["healthy"] * 10 + ["lung_adenocarcinoma"] * 10,
            "600.0": np.arange(20, dtype=float),
        }
    )

    train_idx, val_idx, test_idx = stratified_split(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    assert len(train_idx) == 14
    assert len(val_idx) == 3
    assert len(test_idx) == 3
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)


def test_diffusion_forward_and_samplers_have_expected_shape() -> None:
    torch.manual_seed(42)
    model = SpectralDiffusionUNet(
        spec_len=16,
        num_classes=2,
        base_channels=8,
        channel_multipliers=(1, 2),
        time_emb_dim=16,
        label_dim=16,
        num_heads=2,
    )
    diffusion = SpectralDiffusion(timesteps=4)
    x = torch.randn(2, 16)
    labels = torch.eye(2)
    t = torch.tensor([0, 3])

    pred = model(x, t, labels)
    ddpm = diffusion.sample(model, (2, 16), labels, x.device, sampler="ddpm")
    ddim = diffusion.sample(model, (2, 16), labels, x.device, sampler="ddim")

    assert pred.shape == x.shape
    assert ddpm.shape == x.shape
    assert ddim.shape == x.shape
    assert torch.isfinite(ddim).all()


def test_generated_count_by_class_uses_training_distribution() -> None:
    train_df = pd.DataFrame(
        {
            "label": ["healthy", "healthy", "lung_adenocarcinoma"],
            "600.0": [0.1, 0.2, 0.3],
        }
    )

    assert generated_count_by_class(train_df, ratio=2.0) == {
        "healthy": 4,
        "lung_adenocarcinoma": 2,
    }


def test_build_augmented_train_does_not_modify_validation_or_test() -> None:
    train_df = pd.DataFrame({"label": ["healthy"], "600.0": [1.0]})
    generated_df = pd.DataFrame({"label": ["healthy"], "600.0": [2.0]})

    combined = build_augmented_train(train_df, generated_df, seed=42)

    assert len(combined) == 2
    assert train_df["600.0"].tolist() == [1.0]


def test_compute_metrics_includes_specificity_and_ece() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.7, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.75
    assert metrics["specificity"] == 0.5
    assert "ece" in metrics
    assert 0.0 <= metrics["ece"] <= 1.0


def test_split_indices_json_roundtrip(tmp_path: Path) -> None:
    payload = {"train": [1, 2], "val": [3], "test": [4]}
    path = tmp_path / "split_indices.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_generated_path_uses_stable_ratio_naming(tmp_path: Path) -> None:
    assert generated_path_for_ratio(tmp_path, 0.5).name == "raman_generated_0_5x.csv"
