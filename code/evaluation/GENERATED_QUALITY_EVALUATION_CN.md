
# 生成光谱质量评估程序说明

本程序用于评估扩散模型生成的 Raman 光谱是否与真实训练集光谱保持一致，对应论文中“生成光谱检测/质量分析”的核心思路：定量相似度、分布一致性和光谱特征保真度。

本项目是将论文方法迁移到肺腺癌 Raman 二分类任务，不能直接声称复现出论文中甲状腺或 SLE 数据集的原始数值。默认真实参考数据使用 `data/splits/train.csv`，避免测试集泄漏。

## 运行命令

```bash
python code/evaluation/evaluate_generated_quality.py
```

也可以显式指定输入输出：

```bash
python code/evaluation/evaluate_generated_quality.py ^
  --real data/splits/train.csv ^
  --generated-dir outputs/generated ^
  --output-dir outputs/reports/generated_quality ^
  --top-peaks 8
```

程序会自动扫描 `outputs/generated/raman_generated_*x.csv`，例如 `raman_generated_0_5x.csv`、`raman_generated_1_0x.csv`、`raman_generated_2_0x.csv`。

## 输出文件

- `outputs/reports/generated_quality/tables/generated_quality_metrics.csv`：生成光谱的定量质量指标。
- `outputs/reports/generated_quality/tables/qq_distribution_metrics.csv`：QQ 图对应的分布一致性指标。
- `outputs/reports/generated_quality/tables/peak_intensity_summary.csv`：关键峰位的强度分布摘要。
- `outputs/reports/generated_quality/figures/*_mean_spectra.png`：真实与生成均值光谱对比图。
- `outputs/reports/generated_quality/figures/*_qq_plot.png`：真实与生成强度分布 QQ 图。
- `outputs/reports/generated_quality/figures/*_peak_violin.png`：关键峰位强度小提琴图。
- `outputs/reports/generated_quality/generated_quality_report.md`：汇总报告。

## 指标含义

- MSE：生成样本与同类最近邻真实样本之间的均方误差，越低表示数值失真越少。
- Cosine similarity：生成样本与同类最近邻真实样本的余弦相似度，越高表示光谱形态越接近。
- Pearson correlation：生成样本与同类最近邻真实样本的线性相关系数，越高表示峰形变化趋势越一致。
- Wasserstein distance：逐 Raman 位移比较真实/生成强度分布差异后取均值，越低表示分布越接近。
- QQ R2 / RMSE：将真实与生成强度分布的分位数配对并线性拟合，R2 越高、RMSE 越低表示统计分布越一致。

## 光谱特征保真度

程序会基于真实训练集的均值光谱自动选择主要峰位。优先使用 `scipy.signal.find_peaks` 识别峰，再按峰强度排序；如果未找到明显峰，则回退到均值强度最高的 Raman 位移点。小提琴图用于比较这些峰位处真实/生成样本的中位数、四分位距和概率密度。
