# 肺腺癌 Raman 光谱扩散增强复现项目

本仓库用于迁移复现论文《A research on applying the diffusion model algorithm for Infrared and Raman spectroscopy data augmentation to improve the accuracy of diseases》的核心方法，并适配当前肺腺癌 Raman 数据：87 条健康组光谱与 93 条肺腺癌光谱。

## 实验目标

复现方法链路：

```text
原始 Raman 光谱 -> 数据整理 -> 宇宙射线去除/S-G 平滑/IA-WPLS 基线校正/L2 归一化 -> 真实数据分类基线 -> 条件扩散模型训练 -> 按类别生成增强光谱 -> 增强后分类评估 -> 结果对比
```

本项目不是复刻论文中的甲状腺或 SLE 数据集，而是将论文方法迁移到 `healthy` vs `lung_adenocarcinoma` 二分类任务。

## 数据要求

原始数据建议放入：

```text
data/raw/
```

目标建模格式为每行一个样本：

```text
label,600.106,600.859,601.612,...
healthy,...
lung_adenocarcinoma,...
```

如果原始 Excel 是“列为样本、行为 Raman 位移/波数”，需要先转置为上述格式。类别名固定为：

- `healthy`
- `lung_adenocarcinoma`

验收目标：共 180 条样本，其中健康组 87 条，肺腺癌组 93 条。若原始 Excel 重新核验得到不同分布，应以原始文件为准并同步更新配置。

## 项目结构

```text
configs/                         # 配置文件：数据、扩散模型、分类器参数
data/
  raw/                            # 原始数据，只读保留
  processed/                      # 预处理后的建模矩阵
  splits/                         # 固定随机种子的训练/验证/测试索引
notebooks/                        # 探索分析和可视化
outputs/
  checkpoints/                    # 模型权重
  generated/                      # 扩散模型生成的合成光谱
  reports/                        # 指标表、混淆矩阵、光谱图、PCA/t-SNE
code/                             # 所有实验代码，按论文流程分模块
  data/                           # 数据读取、预处理、划分
  diffusion/                      # 扩散模型、DDPM/DDIM、训练和生成
  classification/                 # SVM/RF/MLP/1D-CNN 分类实验
  evaluation/                     # 指标、画图、报告
  utils/                          # 配置读取、路径、随机种子
scripts/                          # 兼容旧命令的快捷入口，实际调用 code/
tests/                            # 单元测试
真实数据检测/                     # 独立的数据可靠性检查实验，例如 1D-CNN 真实数据分类
```

## 推荐运行顺序

```bash
python code/prepare_data.py --config configs/dataset.yaml
python code/classification/train_baseline.py --config configs/classification.yaml
python code/diffusion/train_diffusion.py --config configs/diffusion.yaml
python code/diffusion/generate_spectra.py --config configs/diffusion.yaml --ratio 0.5
python code/diffusion/generate_spectra.py --config configs/diffusion.yaml --ratio 1.0
python code/diffusion/generate_spectra.py --config configs/diffusion.yaml --ratio 2.0
python code/classification/evaluate_augmented.py --config configs/classification.yaml
python code/evaluation/make_report.py
```

如果想一键运行完整流程，可以使用：

```bash
python code/run_all.py
```

注意：扩散训练默认 `10000` epoch，CPU 环境会很慢。调试时可临时降低 `configs/diffusion.yaml` 中的 `training.epochs` 和 `noise_schedule.timesteps`，正式实验再恢复论文迁移设置。

## 实验原则

- 测试集必须只包含真实样本。
- 扩散模型只能使用训练集训练，不能接触测试集。
- 生成样本只能加入训练集。
- 所有实验固定随机种子并保存划分索引。
- 不默认生成越多越好，需要比较 `0.5x`、`1.0x`、`2.0x`。
- 除分类指标外，还要检查生成光谱形态、Pearson 相似度、Wasserstein 距离、PCA 分布。
- 论文报告的是甲状腺/SLE 数据结果；本项目报告的是肺腺癌迁移任务结果，不能直接声称复现出论文原数值。
