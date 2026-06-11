# DON-PBE：结晶过程时序预测（DeepONet 多任务）

用 **DeepONet** 学习结晶过程的时序算子：**输入前 15 分钟**的过程信息（温度、浓度、粒度分布演化），**同时预测后 5 分钟**的**粒度分布（PSD）与浓度**演化。

- 框架：**PyTorch + CUDA**（适配 **RTX 4060**；TensorFlow 在 Windows 原生不支持新版 GPU，故弃用）
- 硬件目标：RTX 4060 Laptop GPU + Intel i9-13900H
- 多任务输出：PSD `n(L, τ)` + 浓度 `C(τ)`

---

## 1. 任务与数据

原始数据：15 个工况，每工况 dt=1s、10801 个时间点（0~10800s，即 3 小时），含温度 `Temp_K`、浓度 `Conc`、粒度分布 `psd`(1000 个粒径)。

预处理：沿时间每 3 点抽稀 → dt=3s、3601 点，存为读取极快的 `.npz`。

样本切分（滚动滑动窗口，步长 5min）：

```
|<----- 输入 15min (300 点) ----->|<-- 输出 5min (100 点) -->|
 历史 T/C + PSD 历史 n(L,t) + 计划 T     待预测 PSD 时空场 + 浓度时序
（计划 T 训练时取真值，控制时由 rc 规划）

滚动方式（步长 100点=5min）：
  0–15min  → 15–20min
  5–20min  → 20–25min
  10–25min → 25–30min
  ...
```

每个工况按 `window_stride_pts`（默认 100 点=5min）滚动切出多个样本（每工况 33 个）。

**工况划分（随机，共 15 条）**：训练 5 条 / 测试(训练时监控) 7 条 / 推理验证 holdout 3 条（神经网络完全不接触，仅训练后评估）。由 `split_seed` 控制随机性。

---

## 2. 算子网络（DeepONet）输入输出

| 部件 | 内容 | 维度 |
|------|------|------|
| **Branch 输入** | 历史 T ⊕ 历史 C ⊕ **PSD 历史 n(L,t)** ⊕ 输出窗计划 T | 300+300+38400+100 = **39100** |
| **Trunk_PSD** | 查询坐标 `[L, τ]`，τ 为预测窗口相对时间 | `2` |
| **Trunk_Conc** | 查询坐标 `[τ]`，仅输出窗口内时间 | `1` |
| **输出① PSD** | `n(L, τ)`，后 5min 的归一化 PSD（Branch⊙Trunk_PSD）| `n_L_eval × n_out` |
| **输出②浓度** | `C(τ)`，后 5min 的归一化浓度（Branch⊙Trunk_Conc）| `n_out` |

- PSD 与浓度**均为 DeepONet 算子形式**：共享 Branch 编码，各自用 Trunk 查询坐标做内积，不在 Branch 上接 MLP 直接回归浓度。
- PSD 分支：所有窗口共享同一 `(L, τ)` 查询网格；浓度分支共享同一 `τ` 查询网格。
- 训练损失：`PSD_MSE + λ_nonneg·非负惩罚 + λ_conc·浓度_MSE`。

---

## 3. 环境配置（RTX 4060）

> 已用 Python 3.12 + torch 2.5.1+cu121 在 RTX 4060 上验证通过。
> 若 PowerShell 禁止运行 `Activate.ps1`，可**直接用 venv 内的 python**（无需激活），如下所示。

```powershell
# 1) 创建虚拟环境
python -m venv .venv

# 2) 安装 CUDA 版 PyTorch（cu121 适配 Ada 架构的 RTX 4060；约 2.4GB）
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
#   国内可用清华镜像加速：
#   .\.venv\Scripts\python.exe -m pip install torch --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu121

# 3) 安装其余依赖
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 4) 验证 GPU 可用
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

应输出 `True NVIDIA GeForce RTX 4060 ...`。

---

## 4. 使用流程

> 下列命令用 venv 内 python 直调（无需激活）。

```powershell
# 步骤 1：原始 .mat → 抽稀 .npz（仅首次或换 .mat 时运行；Branch 改结构后无需重跑）
.\.venv\Scripts\python.exe scripts\01_preprocess.py

# 步骤 2：训练（多任务：PSD + 浓度；改 Branch 后须重新训练，旧 228 维权重不兼容）
.\.venv\Scripts\python.exe scripts\02_train.py
.\.venv\Scripts\python.exe scripts\02_train.py --epochs 500 --batch 128 --lr 5e-4   # 自定义

# 步骤 3：评估 + 可视化（默认在 holdout 工况上推理，取最新结果目录）
.\.venv\Scripts\python.exe scripts\03_evaluate.py
.\.venv\Scripts\python.exe scripts\03_evaluate.py --case CR_1_13   # 指定 holdout 工况

# 步骤 4：自定义起始时刻的单窗口预测（取该时刻起前15min输入，预测随后5min）
.\.venv\Scripts\python.exe scripts\04_predict.py --case CR_1_13 --start_min 40

# 步骤 6：检查切片数据（均匀取 5 个窗口，含首尾，看 PSD 3D + 温度/浓度）
.\.venv\Scripts\python.exe scripts\06_visualize_slices.py --case CR_1_01

# 步骤 5：全程滚动预测（在整条工况上逐5min滚动，拼出完整预测轨迹）
#   默认在【完整粒径网格】上做全粒度预测，输出全程时空热力图等可视化
.\.venv\Scripts\python.exe scripts\05_rolling_predict.py --case CR_1_13
.\.venv\Scripts\python.exe scripts\05_rolling_predict.py --case CR_1_13 --eval_grid  # 仅用下采样评估网格

# 步骤 10：35min 预热末 15min + NMPC 计划温度 → DON Branch 推理（对照 NMPC）
.\.venv\Scripts\python.exe scripts\10_infer_preheat_planned_T.py
.\.venv\Scripts\python.exe scripts\10_infer_preheat_planned_T.py --run results/<时间戳>
```

> 步骤 5 输出（`results/<run>/rolling/<case>/`）：所有图横轴均从 0min 画起，
> 并以阴影/白虚线标出【输入段(0~15min) → 预测段】分界（输入段为已知仿真值）。
> - `spacetime_full.png` — 全粒度全程 PSD 时空热力图（仿真 / 预测 / 误差，含分界线）
> - `psd_snapshots_full.png` — 全粒度多时刻 PSD 分布对比（含 t=0 输入段时刻）
> - `psd_peak_track.png` — 峰值粒径处 PSD 随时间（含输入段）
> - `conc_full.png` — 全程浓度曲线（含输入段）

> 训练启动时终端会打印：**用了哪些训练工况**、每工况切片数、以及训练/测试/holdout 的**切片总数**。

训练产物在 `results/<时间戳>/`：

```
results/20260610_xxxxxx/
  config.json          超参数与维度记录
  norm_params.npz      归一化与网格参数（评估/部署用）
  loss_curve.png       训练/测试损失曲线
  weights/
    best.pt            最优权重（按测试集 val_loss）
    final.pt           最终权重
    ckpt_epoch_XXXX.pt 定期检查点
  eval/
    psd_<case>.png        后5min 各时刻 PSD 预测 vs 仿真
    spacetime_<case>.png  (L, τ) PSD 时空演化与误差
    conc_<case>.png       后5min 浓度预测 vs 仿真
```

---

## 5. 项目结构

```
DON/
├── README.md
├── requirements.txt
├── donpbe/                  # 核心库（模块化）
│   ├── config.py            # 配置（路径/窗口/网络/训练）
│   ├── device.py            # 设备与随机种子（CUDA/4060）
│   ├── preprocess.py        # mat → npz
│   ├── dataset.py           # 滚动窗口 + Branch/Trunk + 浓度标签 + 随机划分 + 归一化
│   ├── model.py             # DeepONet 双算子 PSD+浓度（PyTorch，多任务）
│   ├── trainer.py           # 训练器（联合损失/AMP/调度/检查点）
│   ├── predictor.py         # 推理封装（单窗口 + 全程滚动预测）
│   └── utils.py             # 指标与可视化（PSD + 浓度）
├── scripts/
│   ├── 01_preprocess.py     # mat → npz
│   ├── 02_train.py          # 训练（启动时打印工况/切片数）
│   ├── 03_evaluate.py       # holdout 评估 + 可视化
│   ├── 04_predict.py        # 自定义起始时刻单窗口预测
│   ├── 05_rolling_predict.py# 全程滚动预测
│   └── 06_visualize_slices.py# 切片数据检查（5 窗口 PSD 3D + T/C）
├── data/                    # 预处理输出的 .npz
└── results/                 # 训练结果
```

---

## 6. 关键可调参数（`donpbe/config.py`）

| 参数 | 含义 | 默认 |
|------|------|------|
| `raw_stride` | 原始序列抽稀步长 | 3 (→dt=3s) |
| `in_seconds` / `out_seconds` | 输入/输出窗口时长(秒) | 900 / 300 |
| `window_stride_pts` | 滚动窗口步长(点) | 100 (=5min) |
| `n_T_sensors` / `n_C_sensors` | 温度/浓度（全分辨率） | 300 / 300 |
| `n_L_sensors` / `n_L_eval` | PSD 输入/输出粒径点数 | 128 / 200 |
| `n_train_cases` / `n_val_cases` / `n_holdout_cases` | 训练/测试/推理验证工况数 | 5 / 7 / 3 |
| `split_seed` | 工况随机划分种子 | 42 |
| `conc_trunk_hiddens` | 浓度 Trunk 隐藏层 | [128, 128] |
| `lambda_nonneg` / `lambda_conc` | 非负/浓度损失权重 | 0.05 / 1.0 |
| `batch_size` / `lr` / `epochs` | 训练超参 | 64 / 1e-3 / 300 |
| `use_amp` | 混合精度 | True |

> 也可在 config 中用 `train_cases` / `val_cases` / `holdout_cases` 显式指定工况名（覆盖随机划分）。
