from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

from utils.publication_figures import (
    CLASS_PALETTE,
    PALETTE,
    apply_publication_style,
    finalize_figure,
    style_axes,
    style_colorbar,
)


CLASS_NAMES = ["Healthy", "Lung adenocarcinoma"]
POSITIVE_CLASS_LABEL = "lung adenocarcinoma"


def safe_name(value: str) -> str:
    return value.replace(".", "_").replace(" ", "_").lower()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    image = ax.imshow(normalized, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, pad=6)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = normalized[row, col]
            text_color = "white" if value > 0.5 else "black"
            ax.text(
                col,
                row,
                f"{matrix[row, col]}\n{value * 100:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Proportion")
    style_colorbar(colorbar)
    finalize_figure(fig, output_path)


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    ax.plot(fpr, tpr, color=PALETTE["blue"], linewidth=1.15, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["gray"], linewidth=0.75, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    finalize_figure(fig, output_path)


def save_probability_distribution(y_true: np.ndarray, y_prob: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.7, 3.0))
    bins = np.linspace(0.0, 1.0, 12)
    ax.hist(
        y_prob[y_true == 0],
        bins=bins,
        color=CLASS_PALETTE["healthy"],
        edgecolor="white",
        linewidth=0.45,
        alpha=0.72,
        label="Healthy",
    )
    ax.hist(
        y_prob[y_true == 1],
        bins=bins,
        color=CLASS_PALETTE["lung_adenocarcinoma"],
        edgecolor="white",
        linewidth=0.45,
        alpha=0.72,
        label="Lung adenocarcinoma",
    )
    ax.axvline(0.5, color=PALETTE["black"], linewidth=0.75, linestyle="--", label="Decision threshold")
    ax.set_xlabel(f"Predicted probability of {POSITIVE_CLASS_LABEL}")
    ax.set_ylabel("Number of spectra")
    ax.set_title(title, pad=6)
    ax.legend(frameon=False)
    style_axes(ax)
    finalize_figure(fig, output_path)


def save_test_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    experiment_name: str,
    figure_dir: Path,
) -> None:
    apply_publication_style()
    prefix = f"{safe_name(model_name)}_{safe_name(experiment_name)}"
    save_confusion_matrix(
        y_true,
        y_pred,
        f"{model_name} {experiment_name} confusion matrix",
        figure_dir / f"{prefix}_confusion_matrix_test.png",
    )
    save_roc_curve(
        y_true,
        y_prob,
        f"{model_name} {experiment_name} ROC curve",
        figure_dir / f"{prefix}_roc_curve_test.png",
    )
    save_probability_distribution(
        y_true,
        y_prob,
        f"{model_name} {experiment_name} probability distribution",
        figure_dir / f"{prefix}_probability_distribution_test.png",
    )

