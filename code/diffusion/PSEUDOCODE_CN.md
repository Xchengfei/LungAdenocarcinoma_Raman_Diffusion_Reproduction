# 光谱扩散模型中文伪代码

本文档根据论文第 3 节和表 2 的“光谱扩散模型算法描述”整理，用于对应本项目 `code/diffusion/` 中的实现。

## 1. 光谱扩散模型整体流程

**输入：**

- 原始光谱 `x0`
- 时间步 `t`
- 类别标签 `label`
- 高斯噪声 `epsilon`

**输出：**

- 模型预测噪声 `epsilon_theta`
- 反向采样得到的生成光谱 `x0_hat`

**过程：**

1. 将一维 Raman 光谱 `x0` padding 到固定长度。
2. 将 padding 后的一维光谱 reshape 为二维矩阵。
3. 对时间步 `t` 做正弦位置编码，再经过 MLP 得到时间嵌入 `t_emb`。
4. 对类别标签 `label` 做 one-hot 编码，再经过 MLP 得到标签嵌入 `label_emb`。
5. 将含噪光谱 `x_t`、时间嵌入和标签嵌入送入条件 U-Net。
6. 编码器逐层下采样，提取不同尺度的光谱结构特征。
7. 瓶颈层使用自注意力建模光谱不同位置之间的全局关系。
8. 使用交叉注意力将类别条件注入瓶颈层和解码器特征。
9. 解码器逐层上采样，并与编码器跳跃连接拼接，恢复峰位、峰强和基线细节。
10. 输出与原始光谱长度相同的预测噪声 `epsilon_theta(x_t, t, label)`。

## 2. 前向扩散过程

**目标：**逐步向真实光谱加入高斯噪声，使其最终接近标准正态分布。

```text
给定真实光谱 x0
给定 beta_start, beta_end, T

构造线性 beta 序列:
    beta_1, beta_2, ..., beta_T

计算:
    alpha_t = 1 - beta_t
    alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t

随机采样时间步 t
随机采样高斯噪声 epsilon ~ N(0, I)

根据闭式公式生成含噪光谱:
    x_t = sqrt(alpha_bar_t) * x0
          + sqrt(1 - alpha_bar_t) * epsilon
```

对应代码：

- `SpectralDiffusion.q_sample`
- `SpectralDiffusion.compute_loss`

## 3. 模型预测噪声过程

**目标：**让模型根据 `x_t`、时间步和类别标签预测被加入的噪声。

```text
输入:
    含噪光谱 x_t
    时间步 t
    类别标签 label

时间编码:
    t_emb = MLP(SinusoidalPositionEmbedding(t))

标签编码:
    label_emb = MLP(one_hot(label))

编码器:
    x1 = Block(x_t, t_emb)
    x2 = Block(MaxPool(x1), t_emb)
    x3 = Block(MaxPool(x2), t_emb)

瓶颈层:
    x_bot = Block(x3, t_emb)
    x_bot = SelfAttention(x_bot)
    x_bot = CrossAttention(x_bot, t_emb)
    x_bot = CrossAttention(x_bot, label_emb)
    x_bot = Block(x_bot, t_emb)

解码器:
    x_up1 = ConvTranspose(x_bot)
    x_up1 = Concat(x_up1, x2)
    x_up1 = Block(x_up1, t_emb)
    x_up1 = CrossAttention(x_up1, label_emb)

    x_up2 = ConvTranspose(x_up1)
    x_up2 = Concat(x_up2, x1)
    x_up2 = Block(x_up2, t_emb)
    x_up2 = CrossAttention(x_up2, label_emb)

输出:
    epsilon_theta = Conv(x_up2)
    epsilon_theta = flatten(epsilon_theta)
    epsilon_theta = epsilon_theta[:原始光谱长度]
```

对应代码：

- `SinusoidalPositionEmbedding`
- `Block`
- `SelfAttention`
- `CrossAttention`
- `SpectralDiffusionUNet.forward`

## 4. 训练流程

**目标：**最小化真实噪声与模型预测噪声之间的均方误差。

```text
重复训练多个 epoch:
    从训练集中采样一个 batch:
        x0, label

    随机采样时间步:
        t ~ Uniform(0, T - 1)

    随机采样真实噪声:
        epsilon ~ N(0, I)

    前向扩散得到含噪光谱:
        x_t = q_sample(x0, t, epsilon)

    模型预测噪声:
        epsilon_theta = model(x_t, t, label)

    计算损失:
        loss = MSE(epsilon_theta, epsilon)

    反向传播并更新模型参数

    在验证集上计算 val_loss
    如果 val_loss 更低:
        保存 diffusion_best.pt
```

对应代码：

- `train_diffusion.py`
- `SpectralDiffusion.compute_loss`

## 5. 反向去噪采样过程

**目标：**从纯高斯噪声出发，在指定类别条件下逐步生成清晰光谱。

```text
给定类别 label
随机采样初始噪声:
    x_T ~ N(0, I)

for t = T - 1, T - 2, ..., 0:
    epsilon_theta = model(x_t, t, label)

    如果使用 DDPM:
        根据后验均值和方差采样 x_{t-1}

    如果使用 DDIM:
        使用隐式确定性更新生成 x_{t-1}

返回:
    x_0_hat
```

对应代码：

- `SpectralDiffusion.p_sample_ddpm`
- `SpectralDiffusion.p_sample_ddim`
- `SpectralDiffusion.sample`

## 6. 条件生成流程

**目标：**按照训练集类别比例生成增强光谱。

```text
读取训练集 train.csv
统计每个类别的样本数
根据增强比例 ratio 计算每类生成数量:
    generated_count = train_count * ratio

加载 diffusion_best.pt
for 每个类别:
    构造该类别的 one-hot 标签
    从随机噪声开始反向采样
    得到该类别生成光谱

合并所有类别的生成光谱
打乱顺序
保存为:
    outputs/generated/raman_generated_{ratio}x.csv
```

对应代码：

- `generate_spectra.py`

## 7. 本项目中如何使用

本节说明如何在当前项目中训练 diffusion 模型，并用训练好的模型生成增强 Raman 光谱。

### 7.1 使用前准备

在运行 diffusion 之前，需要先完成数据预处理，并确保以下文件已经存在：

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
configs/diffusion.yaml
```

其中：

- `train.csv` 用于训练 diffusion 模型。
- `val.csv` 用于选择保存验证损失最低的模型。
- `test.csv` 不用于 diffusion 训练和生成，只保留给后续分类评估。
- `configs/diffusion.yaml` 控制模型结构、扩散步数、训练轮数、生成比例等参数。

### 7.2 训练 diffusion 模型

在项目根目录运行：

```powershell
python code/diffusion/train_diffusion.py --config configs/diffusion.yaml
```

训练脚本会执行以下步骤：

1. 读取 `data/splits/train.csv` 和 `data/splits/val.csv`。
2. 将类别标签转换为 one-hot 条件向量。
3. 构建光谱扩散 U-Net。
4. 随机选择扩散时间步 `t`。
5. 对真实光谱添加高斯噪声，得到 `x_t`。
6. 让模型预测加入的噪声。
7. 使用 MSE 损失优化模型。
8. 在验证集上计算 `val_loss`。
9. 保存验证损失最低的模型。

训练完成后会生成：

```text
outputs/checkpoints/diffusion/diffusion_best.pt
outputs/checkpoints/diffusion/training_history.json
```

其中：

- `diffusion_best.pt` 是后续生成光谱要用的模型权重。
- `training_history.json` 记录每次验证时的训练损失和验证损失。

### 7.3 生成增强光谱

训练完成后，可以按照指定比例生成增强数据。

生成 0.5 倍训练集数量：

```powershell
python code/diffusion/generate_spectra.py --config configs/diffusion.yaml --ratio 0.5
```

生成 1 倍训练集数量：

```powershell
python code/diffusion/generate_spectra.py --config configs/diffusion.yaml --ratio 1.0
```

生成 2 倍训练集数量：

```powershell
python code/diffusion/generate_spectra.py --config configs/diffusion.yaml --ratio 2.0
```

生成脚本会执行以下步骤：

1. 加载 `outputs/checkpoints/diffusion/diffusion_best.pt`。
2. 读取 `data/splits/train.csv`。
3. 统计训练集中每个类别的样本数量。
4. 根据 `ratio` 计算每个类别需要生成多少条光谱。
5. 对每个类别构造 one-hot 条件标签。
6. 从高斯噪声开始进行 DDPM 或 DDIM 反向采样。
7. 得到指定类别的生成光谱。
8. 合并并打乱所有生成样本。
9. 保存为 CSV 文件。

输出文件示例：

```text
outputs/generated/raman_generated_0_5x.csv
outputs/generated/raman_generated_1_0x.csv
outputs/generated/raman_generated_2_0x.csv
```

每个生成 CSV 的格式与训练数据一致：

```text
label, Raman feature 1, Raman feature 2, ..., Raman feature N
```

### 7.4 配置文件中常用参数

主要参数在 `configs/diffusion.yaml` 中修改。

```yaml
noise_schedule:
  timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02

training:
  epochs: 10000
  batch_size: 32
  learning_rate: 0.0001
  validate_every: 100
  checkpoint_dir: outputs/checkpoints/diffusion

generation:
  ratios: [0.5, 1.0, 2.0]
  output_dir: outputs/generated
  sampler: ddim
  eta: 0.0
  batch_size: 32
```

建议：

- 第一次测试代码能否跑通时，可以临时把 `epochs` 改小，例如 `10` 或 `50`。
- 正式训练时再使用较大的 `epochs`。
- `sampler: ddim` 生成速度通常比 `ddpm` 更快。
- `eta: 0.0` 表示确定性 DDIM 采样，生成更稳定。

### 7.5 推荐运行顺序

完整流程建议按下面顺序执行：

```powershell
python code/data/prepare_data.py --config configs/dataset.yaml
python code/diffusion/train_diffusion.py --config configs/diffusion.yaml
python code/diffusion/generate_spectra.py --config configs/diffusion.yaml --ratio 0.5
python code/diffusion/generate_spectra.py --config configs/diffusion.yaml --ratio 1.0
python code/diffusion/generate_spectra.py --config configs/diffusion.yaml --ratio 2.0
```

### 7.6 常见问题

如果提示找不到模型权重：

```text
Missing diffusion checkpoint
```

说明还没有训练 diffusion 模型，或者 `checkpoint_dir` 配置不一致。先运行训练命令。

如果训练很慢：

- diffusion 模型训练本来比 SVM、Random Forest 慢很多。
- 有 GPU 时会自动使用 CUDA。
- 没有 GPU 时会使用 CPU，训练时间会明显变长。
- 可以先减小 `epochs` 或 `timesteps` 做快速测试。

如果生成结果数量不对：

- 生成数量由 `train.csv` 中每个类别数量乘以 `ratio` 得到。
- 例如训练集中 healthy 有 61 条，`ratio=1.0` 时会生成 61 条 healthy。

如果 PowerShell 显示中文乱码：

- 通常是终端编码问题，不代表 Markdown 文件损坏。
- 文件使用 UTF-8 保存，可以在 VS Code、Jupyter 或支持 UTF-8 的编辑器中正常查看。
