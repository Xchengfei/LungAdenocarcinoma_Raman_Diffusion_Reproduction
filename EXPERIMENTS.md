# 复现实验记录

## 数据

- 健康组：87 条 Raman 光谱
- 肺腺癌组：93 条 Raman 光谱
- 类别：`healthy`、`lung_adenocarcinoma`

## 必做对比

| 实验 | 训练数据 | 验证/测试数据 |
|---|---|---|
| real_only | 真实训练集 | 真实验证集 / 真实测试集 |
| diffusion_0_5x | 真实训练集 + 0.5x 生成样本 | 真实验证集 / 真实测试集 |
| diffusion_1_0x | 真实训练集 + 1.0x 生成样本 | 真实验证集 / 真实测试集 |
| diffusion_2_0x | 真实训练集 + 2.0x 生成样本 | 真实验证集 / 真实测试集 |

## 指标

- Accuracy
- Precision
- Recall
- F1-score
- AUC
- Confusion matrix
- Pearson similarity between real and generated spectra
- Wasserstein distance between real and generated spectra
- PCA visualization

## 复现边界

本项目是论文方法在肺腺癌 Raman 二分类数据上的迁移复现。论文中的甲状腺/SLE 数值结果不作为本项目验收目标。
