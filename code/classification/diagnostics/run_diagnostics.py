from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABEL_COLUMN = "label"
LABEL_MAP = {"healthy": 0, "lung_adenocarcinoma": 1}
SEED = 42
N_SPLITS = 5
N_REPEATS = 20
METRICS = ["accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "auc"]
STAGE_FILES = [
    ("raw", "01_raw_spectra.csv"),
    ("cosmic_ray_removed", "02_cosmic_ray_removed.csv"),
    ("savgol_smoothed", "03_savgol_smoothed.csv"),
    ("baseline_corrected", "05_baseline_corrected.csv"),
    ("l2_normalized", "06_l2_normalized.csv"),
]


def project_root() -> Path:
    path = Path(__file__).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "data").exists() and (candidate / "code").exists():
            return candidate
    return Path.cwd()


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.unicode_minus": False,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def load_labeled_csv(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{path} must contain a '{LABEL_COLUMN}' column")

    unknown_labels = sorted(set(df[LABEL_COLUMN]) - set(LABEL_MAP))
    if unknown_labels:
        raise ValueError(f"Unknown labels in {path}: {unknown_labels}")

    feature_columns = [column for column in df.columns if column != LABEL_COLUMN]
    x = df[feature_columns].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].map(LABEL_MAP).to_numpy(dtype=np.int64)
    return df, x, y, feature_columns


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
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


def build_svm() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=SEED,
                ),
            ),
        ]
    )


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def build_pca_svm() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=20, random_state=SEED)),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=SEED,
                ),
            ),
        ]
    )


def build_mlp() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=SEED,
                ),
            ),
        ]
    )


def build_model(model_name: str):
    if model_name == "SVM":
        return build_svm()
    if model_name == "PCA-SVM":
        return build_pca_svm()
    if model_name == "Random Forest":
        return build_random_forest()
    if model_name == "MLP":
        return build_mlp()
    raise ValueError(f"Unsupported model: {model_name}")


def run_repeated_cv(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    feature_columns = [column for column in df.columns if column != LABEL_COLUMN]
    x = df[feature_columns].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].map(LABEL_MAP).to_numpy(dtype=np.int64)

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )
    base_model = build_model(model_name)
    rows = []

    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y), start=1):
        model = clone(base_model)
        model.fit(x[train_index], y[train_index])
        y_pred = model.predict(x[test_index])
        y_prob = model.predict_proba(x[test_index])[:, 1]
        rows.append(
            {
                "model": model_name,
                "fold": fold_index,
                **compute_metrics(y[test_index], y_pred, y_prob),
            }
        )

    return pd.DataFrame(rows)


def summarize_cv_results(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for group_values, group in results.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_columns, group_values))
        row["n_folds"] = int(len(group))
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} ± {std_value:.4f}"


def make_experiment_record_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "Model": row["model"],
                "Accuracy": format_mean_std(row["accuracy_mean"], row["accuracy_std"]),
                "Precision": format_mean_std(row["precision_mean"], row["precision_std"]),
                "Recall/Sensitivity": format_mean_std(row["recall_mean"], row["recall_std"]),
                "Specificity": format_mean_std(row["specificity_mean"], row["specificity_std"]),
                "F1": format_mean_std(row["f1_mean"], row["f1_std"]),
                "AUC": format_mean_std(row["auc_mean"], row["auc_std"]),
                "Folds": int(row["n_folds"]),
            }
        )
    return pd.DataFrame(rows)


def run_fixed_split(split_dir: Path, model_name: str) -> pd.DataFrame:
    _, x_train, y_train, _ = load_labeled_csv(split_dir / "train.csv")
    _, x_val, y_val, _ = load_labeled_csv(split_dir / "val.csv")
    _, x_test, y_test, _ = load_labeled_csv(split_dir / "test.csv")

    val_model = build_model(model_name)
    val_model.fit(x_train, y_train)
    val_pred = val_model.predict(x_val)
    val_prob = val_model.predict_proba(x_val)[:, 1]

    test_model = build_model(model_name)
    x_train_val = np.vstack([x_train, x_val])
    y_train_val = np.concatenate([y_train, y_val])
    test_model.fit(x_train_val, y_train_val)
    test_pred = test_model.predict(x_test)
    test_prob = test_model.predict_proba(x_test)[:, 1]

    rows = [
        {
            "model": model_name,
            "evaluation": "fixed_val_train_on_train",
            "training_samples": int(len(y_train)),
            "evaluation_samples": int(len(y_val)),
            **compute_metrics(y_val, val_pred, val_prob),
        },
        {
            "model": model_name,
            "evaluation": "fixed_test_train_on_train_val",
            "training_samples": int(len(y_train_val)),
            "evaluation_samples": int(len(y_test)),
            **compute_metrics(y_test, test_pred, test_prob),
        },
    ]
    return pd.DataFrame(rows)


def save_bar_plot(summary: pd.DataFrame, output_path: Path) -> None:
    models = summary["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.bar(
        x - width / 2,
        summary["accuracy_mean"],
        width,
        yerr=summary["accuracy_std"],
        color="#9ecae1",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        label="Accuracy",
    )
    ax.bar(
        x + width / 2,
        summary["auc_mean"],
        width,
        yerr=summary["auc_std"],
        color="#3182bd",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        label="AUC",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Cross-validation score")
    ax.set_ylim(0, 1)
    ax.set_title("Repeated cross-validation performance", pad=6)
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_stage_plot(summary: pd.DataFrame, metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for model_name, group in summary.groupby("model"):
        group = group.sort_values("stage_order")
        ax.errorbar(
            group["stage"],
            group[f"{metric}_mean"],
            yerr=group[f"{metric}_std"],
            marker="o",
            linewidth=1.0,
            markersize=3,
            capsize=3,
            label=model_name,
        )
    ax.set_ylabel(metric.upper() if metric == "auc" else metric.capitalize())
    ax.set_xlabel("Preprocessing stage")
    ax.set_ylim(0, 1)
    ax.set_title(f"Performance across preprocessing stages ({metric})", pad=6)
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=25)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_plot_style()
    root = project_root()
    table_dir = root / "outputs" / "reports" / "tables" / "diagnostics"
    figure_dir = root / "outputs" / "reports" / "figures" / "diagnostics"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    model_names = ["SVM", "PCA-SVM", "Random Forest", "MLP"]
    processed_path = root / "data" / "processed" / "raman_spectra_preprocessed.csv"
    processed_df, _, _, _ = load_labeled_csv(processed_path)

    cv_results = pd.concat([run_repeated_cv(processed_df, model_name) for model_name in model_names], ignore_index=True)
    cv_summary = summarize_cv_results(cv_results, ["model"])
    cv_summary.to_csv(table_dir / "cv_model_summary.csv", index=False)
    experiment_record = make_experiment_record_table(cv_summary)
    experiment_record.to_csv(table_dir / "experiment_record_table.csv", index=False)
    save_bar_plot(cv_summary, figure_dir / "cv_accuracy_auc_bar.png")

    fixed_results = pd.concat(
        [run_fixed_split(root / "data" / "splits", model_name) for model_name in model_names],
        ignore_index=True,
    )
    cv_for_comparison = cv_summary[["model", "accuracy_mean", "accuracy_std", "auc_mean", "auc_std"]].copy()
    cv_for_comparison.insert(1, "evaluation", "repeated_cv_mean_std")
    fixed_split_vs_cv = pd.concat([fixed_results, cv_for_comparison], ignore_index=True, sort=False)
    fixed_split_vs_cv.to_csv(table_dir / "fixed_split_vs_cv.csv", index=False)

    stage_rows = []
    data2_dir = root / "data2"
    if data2_dir.exists():
        for stage_order, (stage_name, filename) in enumerate(STAGE_FILES, start=1):
            stage_path = data2_dir / filename
            if not stage_path.exists():
                print(f"Skipping missing preprocessing stage: {stage_path}")
                continue
            stage_df, _, _, _ = load_labeled_csv(stage_path)
            for model_name in model_names:
                result = run_repeated_cv(stage_df, model_name)
                result.insert(0, "stage_order", stage_order)
                result.insert(1, "stage", stage_name)
                stage_rows.append(result)
    else:
        print(f"Skipping preprocessing stage analysis because data2 was not found: {data2_dir}")

    if stage_rows:
        stage_results = pd.concat(stage_rows, ignore_index=True)
        stage_summary = summarize_cv_results(stage_results, ["stage_order", "stage", "model"])
        stage_summary.to_csv(table_dir / "preprocessing_stage_summary.csv", index=False)
        save_stage_plot(stage_summary, "auc", figure_dir / "preprocessing_stage_auc.png")
        save_stage_plot(stage_summary, "accuracy", figure_dir / "preprocessing_stage_accuracy.png")
    else:
        pd.DataFrame().to_csv(table_dir / "preprocessing_stage_summary.csv", index=False)

    summary = {
        "tables": str(table_dir),
        "figures": str(figure_dir),
        "cv_model_summary": str(table_dir / "cv_model_summary.csv"),
        "experiment_record_table": str(table_dir / "experiment_record_table.csv"),
        "fixed_split_vs_cv": str(table_dir / "fixed_split_vs_cv.csv"),
        "preprocessing_stage_summary": str(table_dir / "preprocessing_stage_summary.csv"),
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "models": model_names,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
