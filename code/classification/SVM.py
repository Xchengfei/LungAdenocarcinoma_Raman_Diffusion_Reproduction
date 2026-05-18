from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.publication_figures import (
    CLASS_PALETTE,
    PALETTE,
    apply_publication_style,
    finalize_figure,
    style_axes as publication_style_axes,
    style_colorbar,
)


LABEL_COLUMN = "label"
LABEL_MAP = {"healthy": 0, "lung_adenocarcinoma": 1}
CLASS_NAMES = ["Healthy", "Lung adenocarcinoma"]
POSITIVE_CLASS_LABEL = "lung adenocarcinoma"
SEED = 42
KERNELS = ["linear", "rbf"]
C_VALUES = [0.01, 0.1, 1, 10, 100]
GAMMA_VALUES = ["scale", 0.001, 0.01, 0.1, 1]


def project_root() -> Path:
    path = Path(__file__).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "data").exists() and (candidate / "code").exists():
            return candidate
    return Path.cwd()


def configure_plot_style() -> None:
    apply_publication_style()


def style_axis(ax: plt.Axes) -> None:
    publication_style_axes(ax)


def load_split(split_dir: Path, split_name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = split_dir / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{path} must contain a '{LABEL_COLUMN}' column")

    unknown_labels = sorted(set(df[LABEL_COLUMN]) - set(LABEL_MAP))
    if unknown_labels:
        raise ValueError(f"Unknown labels in {path}: {unknown_labels}")

    feature_columns = [column for column in df.columns if column != LABEL_COLUMN]
    x = df[feature_columns].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].map(LABEL_MAP).to_numpy(dtype=np.int64)
    return df, x, y


def compute_metrics(split: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int | str]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        "split": split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc(fpr, tpr)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_svm(kernel: str, c_value: float, gamma: str | float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=kernel,
                    C=c_value,
                    gamma=gamma,
                    probability=True,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def grid_search_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Pipeline, dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    best_model: Pipeline | None = None
    best_row: dict[str, object] | None = None

    for kernel in KERNELS:
        for c_value in C_VALUES:
            gamma_candidates = ["scale"] if kernel == "linear" else GAMMA_VALUES
            for gamma in gamma_candidates:
                model = build_svm(kernel=kernel, c_value=c_value, gamma=gamma)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_val)
                y_prob = model.predict_proba(x_val)[:, 1]
                metrics = compute_metrics("val", y_val, y_pred, y_prob)
                row = {
                    "kernel": kernel,
                    "C": c_value,
                    "gamma": gamma,
                    **{f"val_{key}": value for key, value in metrics.items() if key != "split"},
                }
                rows.append(row)

                if best_row is None:
                    best_model = model
                    best_row = row
                    continue

                current_score = (float(row["val_auc"]), float(row["val_accuracy"]), float(row["val_f1"]))
                best_score = (
                    float(best_row["val_auc"]),
                    float(best_row["val_accuracy"]),
                    float(best_row["val_f1"]),
                )
                if current_score > best_score:
                    best_model = model
                    best_row = row

    if best_model is None or best_row is None:
        raise RuntimeError("SVM grid search did not train any model")

    return best_model, best_row, pd.DataFrame(rows)


def refit_best_svm(x_train_val: np.ndarray, y_train_val: np.ndarray, best_params: dict[str, object]) -> Pipeline:
    model = build_svm(
        kernel=str(best_params["kernel"]),
        c_value=float(best_params["C"]),
        gamma=best_params["gamma"],
    )
    model.fit(x_train_val, y_train_val)
    return model


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
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
    ax.set_title("SVM confusion matrix (test set)", pad=6)

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
    fig.tight_layout()
    finalize_figure(fig, output_path)
    plt.close(fig)


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    ax.plot(fpr, tpr, color=PALETTE["blue"], linewidth=1.15, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["gray"], linewidth=0.75, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("SVM ROC curve (test set)", pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)
    fig.tight_layout()
    finalize_figure(fig, output_path)
    plt.close(fig)


def save_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
) -> None:
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
    ax.set_title("SVM probability distribution (test set)", pad=6)
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    finalize_figure(fig, output_path)
    plt.close(fig)


def save_pca_predictions(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(x_train)
    coords = pca.transform(x_test)
    correct = y_true == y_pred

    fig, ax = plt.subplots(figsize=(3.7, 3.1))
    markers = {0: "o", 1: "s"}
    for class_id, class_name in enumerate(CLASS_NAMES):
        for is_correct, line_width, suffix in [
            (True, 0.6, "correct"),
            (False, 1.1, "incorrect"),
        ]:
            mask = (y_true == class_id) & (correct == is_correct)
            if not np.any(mask):
                continue
            if is_correct:
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    marker=markers[class_id],
                    facecolors="white",
                    edgecolors=CLASS_PALETTE["healthy"] if class_id == 0 else CLASS_PALETTE["lung_adenocarcinoma"],
                    linewidths=line_width,
                    s=42,
                    label=f"{class_name}, {suffix}",
                )
            else:
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    marker="x",
                    c=PALETTE["black"],
                    linewidths=line_width,
                    s=42,
                    label=f"{class_name}, {suffix}",
                )

    explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title("SVM test predictions in PCA space", pad=6)
    ax.legend(frameon=False, loc="best")
    style_axis(ax)
    fig.tight_layout()
    finalize_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    configure_plot_style()
    root = project_root()
    split_dir = root / "data" / "splits"
    table_dir = root / "outputs" / "reports" / "tables"
    figure_dir = root / "outputs" / "reports" / "figures" / "svm"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    _, x_train, y_train = load_split(split_dir, "train")
    _, x_val, y_val = load_split(split_dir, "val")
    _, x_test, y_test = load_split(split_dir, "test")

    selection_model, best_params, grid_results = grid_search_svm(x_train, y_train, x_val, y_val)
    grid_path = table_dir / "svm_grid_search.csv"
    grid_results.to_csv(grid_path, index=False)

    x_train_val = np.vstack([x_train, x_val])
    y_train_val = np.concatenate([y_train, y_val])
    final_model = refit_best_svm(x_train_val, y_train_val, best_params)

    rows = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for split_name, model, x_split, y_split in [
        ("val", selection_model, x_val, y_val),
        ("test", final_model, x_test, y_test),
    ]:
        y_pred = model.predict(x_split)
        y_prob = model.predict_proba(x_split)[:, 1]
        predictions[split_name] = {"y_pred": y_pred, "y_prob": y_prob}
        rows.append(compute_metrics(split_name, y_split, y_pred, y_prob))

    metrics = pd.DataFrame(rows)
    metrics_path = table_dir / "svm_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    test_pred = predictions["test"]["y_pred"]
    test_prob = predictions["test"]["y_prob"]
    save_confusion_matrix(y_test, test_pred, figure_dir / "svm_confusion_matrix_test.png")
    save_roc_curve(y_test, test_prob, figure_dir / "svm_roc_curve_test.png")
    save_probability_distribution(y_test, test_prob, figure_dir / "svm_probability_distribution_test.png")
    save_pca_predictions(x_train_val, x_test, y_test, test_pred, figure_dir / "svm_pca_test_predictions.png")

    summary = {
        "metrics": str(metrics_path),
        "grid_search": str(grid_path),
        "figures": str(figure_dir),
        "best_params": {
            "kernel": best_params["kernel"],
            "C": best_params["C"],
            "gamma": best_params["gamma"],
            "val_auc": best_params["val_auc"],
            "val_accuracy": best_params["val_accuracy"],
            "val_f1": best_params["val_f1"],
            "final_training_samples": int(len(y_train_val)),
        },
        "test_metrics": metrics[metrics["split"] == "test"].iloc[0].to_dict(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
