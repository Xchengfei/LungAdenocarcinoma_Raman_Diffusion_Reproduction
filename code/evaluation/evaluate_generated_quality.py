from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.prepare_data import LABEL_COLUMN, feature_columns
from utils.config import project_root, resolve_path
from utils.publication_figures import (
    CLASS_PALETTE,
    PALETTE,
    apply_publication_style,
    finalize_figure,
    style_axes,
)


def parse_ratio(path: Path) -> str:
    return path.stem.replace("raman_generated_", "").replace("x", "").replace("_", ".")


def as_float_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return df[columns].to_numpy(dtype=float)


def nearest_neighbor_quality(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    if len(real) == 0 or len(generated) == 0:
        return {
            "mse_nearest_mean": float("nan"),
            "cosine_nearest_mean": float("nan"),
            "pearson_nearest_mean": float("nan"),
        }

    cosine = cosine_similarity(generated, real)
    nearest_indices = np.nanargmax(cosine, axis=1)
    nearest_real = real[nearest_indices]
    mse = np.mean((generated - nearest_real) ** 2, axis=1)

    pearson_values = []
    for sample, match in zip(generated, nearest_real):
        if np.std(sample) == 0 or np.std(match) == 0:
            pearson_values.append(float("nan"))
        else:
            pearson_values.append(float(np.corrcoef(sample, match)[0, 1]))

    return {
        "mse_nearest_mean": float(np.nanmean(mse)),
        "cosine_nearest_mean": float(np.nanmean(np.max(cosine, axis=1))),
        "pearson_nearest_mean": float(np.nanmean(pearson_values)),
    }


def mean_wasserstein(real: np.ndarray, generated: np.ndarray) -> float:
    if len(real) == 0 or len(generated) == 0:
        return float("nan")
    return float(
        np.mean(
            [
                wasserstein_distance(real[:, index], generated[:, index])
                for index in range(real.shape[1])
            ]
        )
    )


def qq_distribution_metrics(real: np.ndarray, generated: np.ndarray, quantile_count: int = 101) -> dict[str, float]:
    if real.size == 0 or generated.size == 0:
        return {
            "qq_r2": float("nan"),
            "qq_rmse": float("nan"),
            "qq_slope": float("nan"),
            "qq_intercept": float("nan"),
        }

    quantiles = np.linspace(0.0, 1.0, quantile_count)
    real_q = np.quantile(real.ravel(), quantiles)
    gen_q = np.quantile(generated.ravel(), quantiles)
    slope, intercept = np.polyfit(real_q, gen_q, deg=1)
    predicted = slope * real_q + intercept
    residual = gen_q - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((gen_q - np.mean(gen_q)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "qq_r2": float(r2),
        "qq_rmse": rmse,
        "qq_slope": float(slope),
        "qq_intercept": float(intercept),
    }


def select_peak_columns(real_df: pd.DataFrame, columns: list[str], top_peaks: int) -> list[str]:
    if top_peaks <= 0:
        return []

    mean_spectrum = real_df[columns].to_numpy(dtype=float).mean(axis=0)
    peak_indices, _ = find_peaks(mean_spectrum)
    if len(peak_indices):
        ranked = peak_indices[np.argsort(mean_spectrum[peak_indices])[::-1]]
    else:
        ranked = np.array([], dtype=int)

    fallback = np.argsort(mean_spectrum)[::-1]
    ordered: list[int] = []
    for index in np.concatenate([ranked, fallback]):
        candidate = int(index)
        if candidate not in ordered:
            ordered.append(candidate)
        if len(ordered) == min(top_peaks, len(columns)):
            break
    return [columns[index] for index in sorted(ordered, key=lambda idx: float(columns[idx]))]


def evaluate_metrics(real_df: pd.DataFrame, generated_df: pd.DataFrame, columns: list[str], ratio: str) -> pd.DataFrame:
    rows = []
    for label in sorted(real_df[LABEL_COLUMN].unique()):
        real = as_float_matrix(real_df[real_df[LABEL_COLUMN] == label], columns)
        generated = as_float_matrix(generated_df[generated_df[LABEL_COLUMN] == label], columns)
        if len(generated) == 0:
            continue
        quality = nearest_neighbor_quality(real, generated)
        quality["wasserstein_mean"] = mean_wasserstein(real, generated)
        rows.append(
            {
                "ratio": ratio,
                "label": label,
                "real_count": len(real),
                "generated_count": len(generated),
                **quality,
            }
        )
    return pd.DataFrame(rows)


def evaluate_qq(real_df: pd.DataFrame, generated_df: pd.DataFrame, columns: list[str], ratio: str) -> pd.DataFrame:
    rows = []
    for label in sorted(real_df[LABEL_COLUMN].unique()):
        real = as_float_matrix(real_df[real_df[LABEL_COLUMN] == label], columns)
        generated = as_float_matrix(generated_df[generated_df[LABEL_COLUMN] == label], columns)
        if len(generated) == 0:
            continue
        rows.append({"ratio": ratio, "label": label, **qq_distribution_metrics(real, generated)})
    return pd.DataFrame(rows)


def summarize_peak_intensities(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    peak_columns: list[str],
    ratio: str,
) -> pd.DataFrame:
    rows = []
    for label in sorted(real_df[LABEL_COLUMN].unique()):
        for source, df in [("real", real_df), ("generated", generated_df)]:
            label_df = df[df[LABEL_COLUMN] == label]
            for peak in peak_columns:
                values = label_df[peak].to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                rows.append(
                    {
                        "ratio": ratio,
                        "label": label,
                        "source": source,
                        "peak": peak,
                        "count": len(values),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "q1": float(np.quantile(values, 0.25)),
                        "q3": float(np.quantile(values, 0.75)),
                    }
                )
    return pd.DataFrame(rows)


def plot_mean_spectra(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    columns: list[str],
    ratio: str,
    output_path: Path,
) -> None:
    x_axis = np.array([float(column) for column in columns])
    labels = sorted(real_df[LABEL_COLUMN].unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(3.25 * len(labels), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels):
        real = as_float_matrix(real_df[real_df[LABEL_COLUMN] == label], columns)
        generated = as_float_matrix(generated_df[generated_df[LABEL_COLUMN] == label], columns)
        ax.plot(x_axis, real.mean(axis=0), label="Real train", color=PALETTE["blue"], linewidth=1.0)
        if len(generated):
            ax.plot(x_axis, generated.mean(axis=0), label="Generated", color=PALETTE["orange"], linewidth=1.0)
        ax.set_title(f"{label} ({ratio}x)")
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.legend(frameon=False)
        style_axes(ax, grid=True)
    axes[0].set_ylabel("Normalized intensity")
    finalize_figure(fig, output_path)


def plot_qq(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    columns: list[str],
    ratio: str,
    output_path: Path,
) -> None:
    labels = sorted(real_df[LABEL_COLUMN].unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(3.05 * len(labels), 2.8))
    axes = np.atleast_1d(axes)
    quantiles = np.linspace(0.0, 1.0, 101)
    for ax, label in zip(axes, labels):
        real = as_float_matrix(real_df[real_df[LABEL_COLUMN] == label], columns)
        generated = as_float_matrix(generated_df[generated_df[LABEL_COLUMN] == label], columns)
        if len(generated) == 0:
            ax.set_visible(False)
            continue
        real_q = np.quantile(real.ravel(), quantiles)
        gen_q = np.quantile(generated.ravel(), quantiles)
        metrics = qq_distribution_metrics(real, generated)
        low = min(float(real_q.min()), float(gen_q.min()))
        high = max(float(real_q.max()), float(gen_q.max()))
        ax.scatter(real_q, gen_q, s=10, alpha=0.72, color=PALETTE["green"], edgecolors="none")
        ax.plot([low, high], [low, high], linestyle="--", color=PALETTE["black"], linewidth=0.8, label="y=x")
        ax.set_title(f"{label} R2={metrics['qq_r2']:.3f}, RMSE={metrics['qq_rmse']:.3f}")
        ax.set_xlabel("Real quantiles")
        ax.set_ylabel("Generated quantiles")
        ax.legend(frameon=False)
        style_axes(ax, grid=True)
    fig.suptitle(f"QQ distribution ({ratio}x)", y=1.02, fontsize=8)
    finalize_figure(fig, output_path)


def plot_peak_violin(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    peak_columns: list[str],
    ratio: str,
    output_path: Path,
) -> None:
    if not peak_columns:
        return
    frames = []
    for source, df in [("real", real_df), ("generated", generated_df)]:
        melted = df[[LABEL_COLUMN, *peak_columns]].melt(
            id_vars=LABEL_COLUMN,
            value_vars=peak_columns,
            var_name="peak",
            value_name="intensity",
        )
        melted["source"] = source
        frames.append(melted)
    plot_df = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(max(5.8, 0.72 * len(peak_columns)), 3.2))
    sns.violinplot(
        data=plot_df,
        x="peak",
        y="intensity",
        hue="source",
        split=True,
        inner="quart",
        cut=0,
        palette={"real": CLASS_PALETTE["healthy"], "generated": CLASS_PALETTE["lung_adenocarcinoma"]},
        linewidth=0.6,
        ax=ax,
    )
    ax.set_title(f"Peak intensity distribution: {ratio}x")
    ax.set_xlabel("Raman peak")
    ax.set_ylabel("Normalized intensity")
    ax.tick_params(axis="x", rotation=45)
    sns.move_legend(ax, "best", frameon=False, title=None)
    style_axes(ax, grid=True)
    finalize_figure(fig, output_path)


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    qq_metrics: pd.DataFrame,
    peak_summary: pd.DataFrame,
    peak_columns: list[str],
) -> Path:
    report_path = output_dir / "generated_quality_report.md"
    lines = [
        "# 生成光谱质量评估报告",
        "",
        "本报告用于评估扩散模型生成 Raman 光谱与真实训练集光谱的一致性。",
        "真实参考数据默认来自训练集，避免将测试集信息用于生成质量评估。",
        "",
        "## 输出文件",
        "- `tables/generated_quality_metrics.csv`: 最近邻 MSE、余弦相似度、Pearson 相关系数、Wasserstein 距离。",
        "- `tables/qq_distribution_metrics.csv`: QQ 图线性拟合的 R2、RMSE、斜率和截距。",
        "- `tables/peak_intensity_summary.csv`: 关键峰位处真实/生成强度分布摘要。",
        "- `figures/`: 均值光谱图、QQ 图和峰强度小提琴图。",
        "",
        "## 关键峰位",
        ", ".join(peak_columns) if peak_columns else "未选择峰位。",
        "",
        "## 指标概览",
    ]
    if metrics.empty:
        lines.append("未找到可评估的生成光谱。")
    else:
        preview = metrics.round(6).to_markdown(index=False)
        lines.extend(["", preview])
    if not qq_metrics.empty:
        lines.extend(["", "## QQ 分布一致性概览", "", qq_metrics.round(6).to_markdown(index=False)])
    if not peak_summary.empty:
        lines.extend(["", "## 峰强度摘要前 20 行", "", peak_summary.head(20).round(6).to_markdown(index=False)])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def evaluate_generated_quality(
    real_path: Path,
    generated_dir: Path,
    output_dir: Path,
    top_peaks: int,
) -> dict[str, object]:
    real_df = pd.read_csv(real_path)
    columns = feature_columns(real_df)
    generated_paths = sorted(generated_dir.glob("raman_generated_*x.csv"))
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    peak_columns = select_peak_columns(real_df, columns, top_peaks)
    metric_frames = []
    qq_frames = []
    peak_frames = []

    for path in generated_paths:
        generated_df = pd.read_csv(path)
        ratio = parse_ratio(path)
        metric_frames.append(evaluate_metrics(real_df, generated_df, columns, ratio))
        qq_frames.append(evaluate_qq(real_df, generated_df, columns, ratio))
        peak_frames.append(summarize_peak_intensities(real_df, generated_df, peak_columns, ratio))
        plot_mean_spectra(real_df, generated_df, columns, ratio, figure_dir / f"{path.stem}_mean_spectra.png")
        plot_qq(real_df, generated_df, columns, ratio, figure_dir / f"{path.stem}_qq_plot.png")
        plot_peak_violin(real_df, generated_df, peak_columns, ratio, figure_dir / f"{path.stem}_peak_violin.png")

    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    qq_metrics = pd.concat(qq_frames, ignore_index=True) if qq_frames else pd.DataFrame()
    peak_summary = pd.concat(peak_frames, ignore_index=True) if peak_frames else pd.DataFrame()

    metrics_path = table_dir / "generated_quality_metrics.csv"
    qq_path = table_dir / "qq_distribution_metrics.csv"
    peak_path = table_dir / "peak_intensity_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    qq_metrics.to_csv(qq_path, index=False)
    peak_summary.to_csv(peak_path, index=False)
    report_path = write_report(output_dir, metrics, qq_metrics, peak_summary, peak_columns)

    return {
        "report": str(report_path),
        "metrics": str(metrics_path),
        "qq_metrics": str(qq_path),
        "peak_summary": str(peak_path),
        "generated_files": len(generated_paths),
        "quality_rows": len(metrics),
        "peak_columns": peak_columns,
    }


def main() -> None:
    apply_publication_style()
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate generated Raman spectra quality.")
    parser.add_argument("--real", default="data/splits/train.csv", help="Real spectra CSV used as reference.")
    parser.add_argument("--generated-dir", default="outputs/generated", help="Directory containing raman_generated_*x.csv.")
    parser.add_argument("--output-dir", default="outputs/reports/generated_quality", help="Directory for tables and figures.")
    parser.add_argument("--top-peaks", type=int, default=8, help="Number of Raman peaks for violin plots.")
    args = parser.parse_args()

    summary = evaluate_generated_quality(
        real_path=resolve_path(args.real, root),
        generated_dir=resolve_path(args.generated_dir, root),
        output_dir=resolve_path(args.output_dir, root),
        top_peaks=args.top_peaks,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
