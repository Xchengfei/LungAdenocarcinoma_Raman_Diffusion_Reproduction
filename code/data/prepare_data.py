from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import ensure_parent, load_yaml, project_root, resolve_path, set_seed


LABEL_COLUMN = "label"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column != LABEL_COLUMN]


def read_raw_spectra(config: dict, root: Path) -> pd.DataFrame:
    raw_cfg = config.get("raw_data", {})
    path = resolve_path(raw_cfg.get("path", "data/raw/raman_spectra.csv"), root)
    if not path.exists():
        fallback_paths = [
            root / "data" / "raw" / "raman_spectra.csv",
            root / "data" / "raw" / "raman_spectra_preprocessed.csv",
            root / "data" / "augmented" / "raman_spectra_preprocessed.csv",
        ]
        path = next((candidate for candidate in fallback_paths if candidate.exists()), path)

    if not path.exists():
        raise FileNotFoundError(f"Raw spectra file not found: {path}")

    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)

    healthy = pd.read_excel(path, sheet_name=raw_cfg.get("healthy_sheet", "Healthy"), index_col=0)
    cancer = pd.read_excel(path, sheet_name=raw_cfg.get("cancer_sheet", "Cancer"), index_col=0)
    cancer.index = healthy.index
    healthy_t = healthy.transpose()
    cancer_t = cancer.transpose()
    healthy_t.columns = [str(column) for column in healthy_t.columns]
    cancer_t.columns = [str(column) for column in cancer_t.columns]
    healthy_t.insert(0, LABEL_COLUMN, "healthy")
    cancer_t.insert(0, LABEL_COLUMN, "lung_adenocarcinoma")
    return pd.concat([healthy_t, cancer_t], ignore_index=True)


def whittaker_smooth(y: np.ndarray, weights: np.ndarray, lam: float, order: int) -> np.ndarray:
    size = len(y)
    weight_matrix = sparse.diags(weights, 0, shape=(size, size))
    difference = sparse.eye(size, format="csc")
    for _ in range(order):
        difference = difference[1:] - difference[:-1]
    return spsolve(weight_matrix + lam * difference.T @ difference, weights * y)


def iawpls_baseline(
    spectrum: np.ndarray,
    lam: float = 1e6,
    order: int = 3,
    iterations: int = 100,
    tolerance: float = 0.001,
    smooth_window: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    y = spectrum.astype(float).copy()
    weights = np.ones(len(y), dtype=float)
    kernel = np.ones(smooth_window, dtype=float) / smooth_window

    for _ in range(iterations):
        baseline = whittaker_smooth(y, weights, lam, order)
        residual = y - baseline
        new_weights = np.where(residual < 0, 1.0, 0.0)
        smoothed = np.convolve(new_weights, kernel, mode="same")
        if np.max(np.abs(smoothed - weights)) < tolerance:
            weights = smoothed
            break
        weights = smoothed

    baseline = whittaker_smooth(y, weights, lam, order)
    return baseline, y - baseline


def remove_cosmic_rays(
    spectrum: np.ndarray, kernel_size: int = 7, threshold_factor: float = 6.0
) -> np.ndarray:
    median = median_filter(spectrum, size=kernel_size)
    diff = spectrum - median
    mad = np.median(np.abs(diff))
    if mad == 0:
        return spectrum
    cleaned = spectrum.copy()
    spikes = diff > threshold_factor * mad
    cleaned[spikes] = median[spikes]
    return cleaned


def preprocess_spectra(df: pd.DataFrame, preprocessing_config: dict) -> pd.DataFrame:
    columns = feature_columns(df)
    spectra = df[columns].to_numpy(dtype=float)
    processed = np.zeros_like(spectra, dtype=float)

    smoothing_window = int(preprocessing_config.get("smoothing_window", 9))
    if smoothing_window % 2 == 0:
        smoothing_window += 1
    smoothing_polyorder = int(preprocessing_config.get("smoothing_polyorder", 3))
    order = int(preprocessing_config.get("baseline_poly_order", 3))
    iterations = int(preprocessing_config.get("baseline_iterations", 100))
    tolerance = float(preprocessing_config.get("baseline_tolerance", 0.001))
    smooth_window = int(preprocessing_config.get("baseline_smooth_window", 9))

    for index, spectrum in enumerate(spectra):
        cleaned = remove_cosmic_rays(spectrum)
        smoothed = savgol_filter(cleaned, window_length=smoothing_window, polyorder=smoothing_polyorder)
        _, corrected = iawpls_baseline(
            smoothed, order=order, iterations=iterations, tolerance=tolerance, smooth_window=smooth_window
        )
        norm = np.linalg.norm(corrected)
        processed[index] = corrected / norm if norm > 0 else corrected

    result = df.copy()
    result[columns] = processed
    return result


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    indices = np.arange(len(df))
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df[LABEL_COLUMN],
    )
    val_fraction_of_temp = val_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_fraction_of_temp,
        random_state=seed,
        stratify=df.iloc[temp_idx][LABEL_COLUMN],
    )
    return sorted(train_idx.tolist()), sorted(val_idx.tolist()), sorted(test_idx.tolist())


def save_split_csvs(df: pd.DataFrame, indices: dict[str, Iterable[int]], split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in indices.items():
        df.iloc[list(rows)].reset_index(drop=True).to_csv(split_dir / f"{name}.csv", index=False)


def save_npz(df: pd.DataFrame, path: Path, class_map: dict[str, int]) -> None:
    columns = feature_columns(df)
    labels = df[LABEL_COLUMN].map(class_map).to_numpy(dtype=np.int64)
    spectra = df[columns].to_numpy(dtype=np.float32)
    ensure_parent(path)
    np.savez_compressed(path, spectra=spectra, labels=labels, feature_names=np.array(columns))


def prepare_dataset(config_path: str | Path) -> dict[str, object]:
    root = project_root()
    config = load_yaml(config_path)
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    raw_df = read_raw_spectra(config, root)
    if LABEL_COLUMN not in raw_df.columns:
        raise ValueError(f"Input spectra must include a {LABEL_COLUMN!r} column")
    raw_df = raw_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    processed_df = preprocess_spectra(raw_df, config.get("preprocessing", {}))

    outputs = config.get("outputs", {})
    processed_csv = resolve_path(outputs.get("processed_csv", "data/processed/raman_spectra_preprocessed.csv"), root)
    processed_npz = resolve_path(outputs.get("processed_npz", "data/processed/processed_spectra.npz"), root)
    split_json = resolve_path(outputs.get("split_indices", "data/splits/split_indices.json"), root)
    split_dir = split_json.parent

    ensure_parent(processed_csv)
    processed_df.to_csv(processed_csv, index=False)
    save_npz(processed_df, processed_npz, config.get("classes", {"healthy": 0, "lung_adenocarcinoma": 1}))

    split_cfg = config.get("split", {})
    train_idx, val_idx, test_idx = stratified_split(
        processed_df,
        float(split_cfg.get("train_ratio", 0.7)),
        float(split_cfg.get("val_ratio", 0.15)),
        float(split_cfg.get("test_ratio", 0.15)),
        seed,
    )
    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    ensure_parent(split_json)
    split_json.write_text(json.dumps(split_indices, ensure_ascii=False, indent=2), encoding="utf-8")
    save_split_csvs(processed_df, split_indices, split_dir)

    summary = {
        "processed_csv": str(processed_csv),
        "processed_npz": str(processed_npz),
        "split_indices": str(split_json),
        "counts": processed_df[LABEL_COLUMN].value_counts().to_dict(),
        "splits": {name: len(rows) for name, rows in split_indices.items()},
    }
    summary_path = split_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: preprocess raw Raman data and create splits.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
