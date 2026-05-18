import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from classification.evaluate_augmented import build_augmented_train, compute_metrics, generated_path_for_ratio
from augmented_classification.data_builder import build_experiment_datasets
from data.prepare_data import stratified_split
from diffusion.diffusion_process import SpectralDiffusion
from diffusion.generate_spectra import generated_count_by_class
from diffusion.model import SpectralDiffusionUNet
from evaluation.evaluate_generated_quality import (
    mean_wasserstein,
    nearest_neighbor_quality,
    qq_distribution_metrics,
    select_peak_columns,
)


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


def test_augmented_datasets_keep_validation_and_test_real_only(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    generated_dir = tmp_path / "generated"
    split_dir.mkdir()
    generated_dir.mkdir()
    train = pd.DataFrame({"label": ["healthy", "lung_adenocarcinoma"], "600.0": [1.0, 2.0]})
    val = pd.DataFrame({"label": ["healthy"], "600.0": [3.0]})
    test = pd.DataFrame({"label": ["lung_adenocarcinoma"], "600.0": [4.0]})
    generated = pd.DataFrame({"label": ["healthy"], "600.0": [9.0]})
    train.to_csv(split_dir / "train.csv", index=False)
    val.to_csv(split_dir / "val.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)
    generated.to_csv(generated_dir / "raman_generated_1_0x.csv", index=False)

    datasets, warnings = build_experiment_datasets(
        split_dir,
        generated_dir,
        [{"name": "diffusion_1_0x", "augmentation_ratio": 1.0}],
        seed=42,
    )

    assert warnings == []
    assert len(datasets) == 1
    assert len(datasets[0].train_df) == 3
    assert datasets[0].train_generated_count == 1
    assert datasets[0].val_df["600.0"].tolist() == [3.0]
    assert datasets[0].test_df["600.0"].tolist() == [4.0]


def test_missing_generated_file_reports_warning(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    generated_dir = tmp_path / "generated"
    split_dir.mkdir()
    generated_dir.mkdir()
    split = pd.DataFrame({"label": ["healthy", "lung_adenocarcinoma"], "600.0": [1.0, 2.0]})
    split.to_csv(split_dir / "train.csv", index=False)
    split.to_csv(split_dir / "val.csv", index=False)
    split.to_csv(split_dir / "test.csv", index=False)

    datasets, warnings = build_experiment_datasets(
        split_dir,
        generated_dir,
        [{"name": "diffusion_2_0x", "augmentation_ratio": 2.0}],
        seed=42,
    )

    assert datasets == []
    assert len(warnings) == 1
    assert "Missing generated file" in warnings[0]


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


def test_generated_quality_metrics_are_finite_with_unequal_counts() -> None:
    real = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [1.0, 0.0, 0.0],
        ]
    )
    generated = np.array(
        [
            [0.0, 0.95, 0.05],
            [0.9, 0.1, 0.0],
        ]
    )

    metrics = nearest_neighbor_quality(real, generated)
    wasserstein = mean_wasserstein(real, generated)

    assert np.isfinite(metrics["mse_nearest_mean"])
    assert np.isfinite(metrics["cosine_nearest_mean"])
    assert np.isfinite(metrics["pearson_nearest_mean"])
    assert np.isfinite(wasserstein)
    assert metrics["cosine_nearest_mean"] <= 1.0


def test_qq_distribution_metrics_include_r2_and_rmse() -> None:
    real = np.array([[0.0, 0.5, 1.0], [0.1, 0.6, 1.1]])
    generated = real + 0.01

    metrics = qq_distribution_metrics(real, generated)

    assert "qq_r2" in metrics
    assert "qq_rmse" in metrics
    assert np.isfinite(metrics["qq_r2"])
    assert np.isfinite(metrics["qq_rmse"])


def test_select_peak_columns_uses_peaks_and_fallback() -> None:
    peaked = pd.DataFrame(
        {
            "label": ["healthy", "healthy"],
            "600.0": [0.0, 0.0],
            "601.0": [1.0, 1.1],
            "602.0": [0.0, 0.0],
            "603.0": [0.8, 0.9],
            "604.0": [0.0, 0.0],
        }
    )
    flat = pd.DataFrame(
        {
            "label": ["healthy", "healthy"],
            "600.0": [0.2, 0.2],
            "601.0": [0.2, 0.2],
            "602.0": [0.2, 0.2],
        }
    )

    assert select_peak_columns(peaked, ["600.0", "601.0", "602.0", "603.0", "604.0"], 2) == [
        "601.0",
        "603.0",
    ]
    assert len(select_peak_columns(flat, ["600.0", "601.0", "602.0"], 2)) == 2
