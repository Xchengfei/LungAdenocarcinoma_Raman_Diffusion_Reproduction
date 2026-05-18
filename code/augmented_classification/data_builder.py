from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.prepare_data import LABEL_COLUMN, feature_columns


LABEL_MAP = {"healthy": 0, "lung_adenocarcinoma": 1}


@dataclass(frozen=True)
class ExperimentDataset:
    name: str
    augmentation_ratio: float
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_columns: list[str]
    train_real_count: int
    train_generated_count: int
    generated_path: Path | None


def generated_path_for_ratio(generated_dir: Path, ratio: float) -> Path:
    ratio_label = str(ratio).replace(".", "_")
    return generated_dir / f"raman_generated_{ratio_label}x.csv"


def build_augmented_train(train_df: pd.DataFrame, generated_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    combined = pd.concat([train_df.copy(), generated_df.copy()], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_split(split_dir: Path, split_name: str) -> pd.DataFrame:
    path = split_dir / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path)
    validate_labels(df, path)
    return df


def validate_labels(df: pd.DataFrame, source: Path | str) -> None:
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{source} must contain a {LABEL_COLUMN!r} column")
    unknown = sorted(set(df[LABEL_COLUMN]) - set(LABEL_MAP))
    if unknown:
        raise ValueError(f"Unknown labels in {source}: {unknown}")


def validate_feature_columns(reference_columns: list[str], df: pd.DataFrame, source: Path | str) -> None:
    missing = [column for column in reference_columns if column not in df.columns]
    extra = [column for column in feature_columns(df) if column not in reference_columns]
    if missing or extra:
        raise ValueError(f"Feature columns in {source} do not match training split")


def dataframe_to_xy(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = df[columns].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].map(LABEL_MAP).to_numpy(dtype=np.int64)
    return x, y


def build_experiment_datasets(
    split_dir: Path,
    generated_dir: Path,
    experiments: list[dict[str, object]],
    seed: int,
) -> tuple[list[ExperimentDataset], list[str]]:
    train_df = load_split(split_dir, "train")
    val_df = load_split(split_dir, "val")
    test_df = load_split(split_dir, "test")
    columns = feature_columns(train_df)
    validate_feature_columns(columns, val_df, split_dir / "val.csv")
    validate_feature_columns(columns, test_df, split_dir / "test.csv")

    datasets: list[ExperimentDataset] = []
    warnings: list[str] = []
    for experiment in experiments:
        name = str(experiment["name"])
        ratio = float(experiment.get("augmentation_ratio", 0.0))
        generated_path: Path | None = None
        generated_count = 0
        experiment_train = train_df.copy()

        if ratio > 0:
            generated_path = generated_path_for_ratio(generated_dir, ratio)
            if not generated_path.exists():
                warnings.append(f"Missing generated file for {name}: {generated_path}")
                continue
            generated_df = pd.read_csv(generated_path)
            validate_labels(generated_df, generated_path)
            validate_feature_columns(columns, generated_df, generated_path)
            generated_count = len(generated_df)
            experiment_train = build_augmented_train(train_df, generated_df, seed)

        datasets.append(
            ExperimentDataset(
                name=name,
                augmentation_ratio=ratio,
                train_df=experiment_train,
                val_df=val_df.copy(),
                test_df=test_df.copy(),
                feature_columns=columns,
                train_real_count=len(train_df),
                train_generated_count=generated_count,
                generated_path=generated_path,
            )
        )

    return datasets, warnings

