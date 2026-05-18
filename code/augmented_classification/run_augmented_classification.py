from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from augmented_classification.data_builder import build_experiment_datasets, dataframe_to_xy
from augmented_classification.metrics import compute_metrics
from augmented_classification.models import model_specs, predict_probability, refit_if_needed, select_best_model
from augmented_classification.plots import save_test_figures
from utils.config import load_yaml, project_root, resolve_path, set_seed


DEFAULT_EXPERIMENTS = [
    {"name": "real_only", "augmentation_ratio": 0.0},
    {"name": "diffusion_0_5x", "augmentation_ratio": 0.5},
    {"name": "diffusion_1_0x", "augmentation_ratio": 1.0},
    {"name": "diffusion_2_0x", "augmentation_ratio": 2.0},
]


def write_markdown_report(report_dir: Path, metrics: pd.DataFrame, warnings: list[str]) -> Path:
    report_path = report_dir / "augmented_classification_report.md"
    lines = [
        "# 数据增强后分类实验报告",
        "",
        "本报告比较真实训练集与扩散生成增强训练集在真实验证集和真实测试集上的分类表现。",
        "生成样本只加入训练集，验证集和测试集均保持真实样本。",
        "",
        "## 输出文件",
        "- `tables/augmented_classification_metrics.csv`: 验证集和测试集分类指标。",
        "- `tables/augmented_classification_test_summary.csv`: 测试集汇总表。",
        "- `tables/*_grid_search.csv`: 各实验和模型的验证集参数搜索结果。",
        "- `figures/`: 测试集混淆矩阵、ROC 曲线和概率分布图。",
        "",
        "## 测试集指标概览",
    ]
    test_metrics = metrics[metrics["split"] == "test"].copy()
    if test_metrics.empty:
        lines.append("未生成测试集指标。")
    else:
        preview_columns = [
            "experiment",
            "augmentation_ratio",
            "model",
            "train_generated_count",
            "accuracy",
            "f1",
            "auc",
            "sensitivity",
            "specificity",
        ]
        lines.extend(["", test_metrics[preview_columns].round(6).to_markdown(index=False)])
    if warnings:
        lines.extend(["", "## 警告", *[f"- {warning}" for warning in warnings]])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def evaluate_augmented_classification(config_path: str | Path) -> dict[str, object]:
    root = project_root()
    config = load_yaml(config_path)
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    data_cfg = config.get("data", {})
    split_dir = resolve_path(data_cfg.get("split_dir", "data/splits"), root)
    generated_dir = resolve_path(data_cfg.get("generated_dir", "outputs/generated"), root)
    base_report_dir = resolve_path(config.get("outputs", {}).get("report_dir", "outputs/reports"), root)
    report_dir = base_report_dir / "augmented_classification"
    table_dir = report_dir / "tables"
    figure_dir = report_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    experiments = config.get("experiments", DEFAULT_EXPERIMENTS)
    datasets, warnings = build_experiment_datasets(split_dir, generated_dir, experiments, seed)

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        x_train, y_train = dataframe_to_xy(dataset.train_df, dataset.feature_columns)
        x_val, y_val = dataframe_to_xy(dataset.val_df, dataset.feature_columns)
        x_test, y_test = dataframe_to_xy(dataset.test_df, dataset.feature_columns)
        x_train_val = np.vstack([x_train, x_val])
        y_train_val = np.concatenate([y_train, y_val])
        max_pca_components = min(x_train.shape[0], x_train.shape[1])

        for spec in model_specs(max_pca_components):
            selected_model, best_params, grid_results = select_best_model(spec, x_train, y_train, x_val, y_val, seed)
            grid_path = table_dir / f"{dataset.name}_{spec.name}_grid_search.csv"
            grid_results.to_csv(grid_path, index=False)
            final_model = refit_if_needed(spec, best_params, selected_model, x_train_val, y_train_val, seed)

            for split_name, x_split, y_split in [("val", x_val, y_val), ("test", x_test, y_test)]:
                model_for_split = selected_model if split_name == "val" else final_model
                y_pred = model_for_split.predict(x_split)
                y_prob = predict_probability(model_for_split, x_split)
                row = {
                    "experiment": dataset.name,
                    "augmentation_ratio": dataset.augmentation_ratio,
                    "model": spec.name,
                    "split": split_name,
                    "train_real_count": dataset.train_real_count,
                    "train_generated_count": dataset.train_generated_count,
                    **compute_metrics(y_split, y_pred, y_prob),
                }
                rows.append(row)

                if split_name == "test":
                    save_test_figures(y_split, y_pred, y_prob, spec.name, dataset.name, figure_dir / spec.name)

    metrics = pd.DataFrame(rows)
    metrics_path = table_dir / "augmented_classification_metrics.csv"
    summary_path = table_dir / "augmented_classification_test_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    metrics[metrics["split"] == "test"].to_csv(summary_path, index=False)
    report_path = write_markdown_report(report_dir, metrics, warnings)

    return {
        "metrics": str(metrics_path),
        "test_summary": str(summary_path),
        "report": str(report_path),
        "figures": str(figure_dir),
        "experiments": len(datasets),
        "rows": len(metrics),
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diffusion-augmented Raman classification experiments.")
    parser.add_argument("--config", default="configs/classification.yaml")
    args = parser.parse_args()
    print(json.dumps(evaluate_augmented_classification(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

