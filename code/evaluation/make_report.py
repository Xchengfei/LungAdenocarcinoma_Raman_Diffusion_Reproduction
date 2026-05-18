from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.prepare_data import feature_columns
from utils.config import project_root
from utils.publication_figures import PALETTE, apply_publication_style, finalize_figure, style_axes


def nearest_pearson(real: np.ndarray, generated: np.ndarray) -> float:
    if len(real) == 0 or len(generated) == 0:
        return float("nan")
    correlations = []
    for sample in generated:
        corr = np.corrcoef(real, sample[None, :])[:-1, -1]
        correlations.append(np.nanmax(corr))
    return float(np.nanmean(correlations))


def evaluate_generated_quality(split_dir: Path, generated_dir: Path, table_dir: Path) -> pd.DataFrame:
    train_df = pd.read_csv(split_dir / "train.csv")
    columns = feature_columns(train_df)
    rows = []
    for path in sorted(generated_dir.glob("raman_generated_*x.csv")):
        generated_df = pd.read_csv(path)
        ratio = path.stem.replace("raman_generated_", "").replace("x", "").replace("_", ".")
        for label in sorted(train_df["label"].unique()):
            real = train_df[train_df["label"] == label][columns].to_numpy(dtype=float)
            gen = generated_df[generated_df["label"] == label][columns].to_numpy(dtype=float)
            if len(gen) == 0:
                continue
            rows.append(
                {
                    "ratio": ratio,
                    "label": label,
                    "pearson_nearest_mean": nearest_pearson(real, gen),
                    "wasserstein_mean": float(
                        np.mean(
                            [
                                wasserstein_distance(real[:, index], gen[:, index])
                                for index in range(len(columns))
                            ]
                        )
                    ),
                    "generated_count": len(gen),
                }
            )
    quality = pd.DataFrame(rows)
    output = table_dir / "generated_quality.csv"
    quality.to_csv(output, index=False)
    return quality


def plot_mean_spectra(split_dir: Path, generated_dir: Path, figure_dir: Path) -> None:
    train_df = pd.read_csv(split_dir / "train.csv")
    columns = feature_columns(train_df)
    x_axis = np.array([float(column) for column in columns])
    for path in sorted(generated_dir.glob("raman_generated_*x.csv")):
        generated_df = pd.read_csv(path)
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
        for ax, label in zip(axes, sorted(train_df["label"].unique())):
            real = train_df[train_df["label"] == label][columns].to_numpy(dtype=float)
            gen = generated_df[generated_df["label"] == label][columns].to_numpy(dtype=float)
            ax.plot(x_axis, real.mean(axis=0), label="Real train", color=PALETTE["blue"], linewidth=1.0)
            if len(gen):
                ax.plot(x_axis, gen.mean(axis=0), label="Generated", color=PALETTE["orange"], linewidth=1.0)
            ax.set_title(label)
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.legend(frameon=False)
            style_axes(ax, grid=True)
        axes[0].set_ylabel("Normalized intensity")
        fig.tight_layout()
        finalize_figure(fig, figure_dir / f"{path.stem}_mean_spectra.png")
        plt.close(fig)


def plot_pca(split_dir: Path, generated_dir: Path, figure_dir: Path) -> None:
    train_df = pd.read_csv(split_dir / "train.csv")
    columns = feature_columns(train_df)
    for path in sorted(generated_dir.glob("raman_generated_*x.csv")):
        generated_df = pd.read_csv(path)
        real = train_df[columns].to_numpy(dtype=float)
        gen = generated_df[columns].to_numpy(dtype=float)
        data = np.vstack([real, gen])
        coords = PCA(n_components=2, random_state=42).fit_transform(data)
        source = np.array(["real"] * len(real) + ["generated"] * len(gen))
        labels = pd.concat([train_df["label"], generated_df["label"]], ignore_index=True)
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        palette = {"real": PALETTE["blue"], "generated": PALETTE["orange"]}
        for src in ["real", "generated"]:
            mask = source == src
            ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.72, color=palette[src], edgecolors="none", label=src)
        ax.set_title(f"PCA distribution: {path.stem}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False)
        style_axes(ax)
        fig.tight_layout()
        finalize_figure(fig, figure_dir / f"{path.stem}_pca.png")
        plt.close(fig)


def write_markdown_report(report_dir: Path) -> Path:
    metrics_path = report_dir / "tables" / "classification_metrics.csv"
    quality_path = report_dir / "tables" / "generated_quality.csv"
    output = report_dir / "migration_reproduction_report.md"
    lines = [
        "# Lung adenocarcinoma Raman diffusion migration report",
        "",
        "This project migrates the paper method to a lung adenocarcinoma Raman binary task.",
        "The paper's thyroid/SLE numerical results are not claimed as reproduced here.",
        "",
        "## Outputs",
        f"- Classification metrics: `{metrics_path}`",
        f"- Generated quality metrics: `{quality_path}`",
        "- Figures: `outputs/reports/figures/`",
        "",
        "## Method alignment",
        "- Aligned: IA-WPLS preprocessing, conditional diffusion, time embedding, label conditioning, attention, augmentation-ratio comparison.",
        "- Adapted: lung adenocarcinoma labels/data, independent real-only test set, project-specific classifiers and reporting.",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def make_report() -> dict[str, object]:
    apply_publication_style()
    root = project_root()
    split_dir = root / "data" / "splits"
    generated_dir = root / "outputs" / "generated"
    report_dir = root / "outputs" / "reports"
    table_dir = report_dir / "tables"
    figure_dir = report_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    quality = evaluate_generated_quality(split_dir, generated_dir, table_dir)
    if any(generated_dir.glob("raman_generated_*x.csv")):
        plot_mean_spectra(split_dir, generated_dir, figure_dir)
        plot_pca(split_dir, generated_dir, figure_dir)
    report = write_markdown_report(report_dir)
    return {"report": str(report), "quality_rows": len(quality)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(make_report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
