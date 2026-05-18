# 数据增强后分类程序

本文件夹专门运行“扩散生成 Raman 光谱加入训练集后的分类实验”。验证集和测试集始终只使用真实样本，生成样本只加入训练集。

## 有序步骤

1. 确认已经存在真实数据划分：
   - `data/splits/train.csv`
   - `data/splits/val.csv`
   - `data/splits/test.csv`
2. 确认已经存在扩散生成光谱：
   - `outputs/generated/raman_generated_0_5x.csv`
   - `outputs/generated/raman_generated_1_0x.csv`
   - `outputs/generated/raman_generated_2_0x.csv`
3. 构建实验训练集：
   - `real_only`: 只用真实训练集。
   - `diffusion_0_5x`: 真实训练集 + 0.5x 生成样本。
   - `diffusion_1_0x`: 真实训练集 + 1.0x 生成样本。
   - `diffusion_2_0x`: 真实训练集 + 2.0x 生成样本。
4. 使用真实验证集做模型参数选择。
5. 使用真实测试集报告最终分类指标。
6. 输出指标表、参数搜索表、混淆矩阵、ROC 曲线、概率分布图和 Markdown 汇总报告。

默认参数搜索使用小规模候选网格，目的是让一键复现实验稳定完成；如需更大范围调参，可以在 `models.py` 中扩展候选参数。

## 运行命令

```bash
python code/augmented_classification/run_augmented_classification.py --config configs/classification.yaml
```

## 输出位置

- `outputs/reports/augmented_classification/tables/augmented_classification_metrics.csv`
- `outputs/reports/augmented_classification/tables/augmented_classification_test_summary.csv`
- `outputs/reports/augmented_classification/tables/*_grid_search.csv`
- `outputs/reports/augmented_classification/figures/`
- `outputs/reports/augmented_classification/augmented_classification_report.md`
