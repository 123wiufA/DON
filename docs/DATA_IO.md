# DON 项目数据说明

本文档描述 **DON**（DeepONet 算子学习）中数据从原始仿真到训练、推理的完整流转：时间轴、粒径网格、Branch/Trunk、标签与输出。

- 网络张量细节：[`BRANCH_TRUNK_IO.md`](BRANCH_TRUNK_IO.md)  
- NMPC 侧对接：[`../../NMPC_Modular/docs/DATA_IO.md`](../../NMPC_Modular/docs/DATA_IO.md)  
- 默认配置：`donpbe/config.py`

---

## 1. 数据流总览

```
Simulation_Data_DeepONet.mat  (dt=1s, 15 工况)
        │  scripts/01_preprocess.py  (stride=3)
        ▼
dataset_3s.npz  (dt=3s, T/C/psd/L/t)
        │  PBEWindowData 滑动切窗 + assemble_branch
        ▼
训练样本: branch(39100), label_psd(20000), label_conc(100)
        │  scripts/02_train.py
        ▼
results/<run>/  (weights, norm_params.npz, config.json)
        │  Predictor 推理
        ▼
输出: 后 5min 的 n(L,τ) 与 C(τ)
```

**算子语义**：过去 15min 的过程信息（T、C、PSD 历史）+ 随后 5min **计划温度** → 预测随后 5min 的 **PSD 时空场** 与 **浓度时序**。

---

## 2. 原始数据（`.mat`）

| 项 | 说明 |
|----|------|
| 路径 | `data/Simulation_Data_DeepONet.mat` |
| 结构 | `Dataset/<工况名>/` |
| 工况数 | 15（`CR_1_00` … `CR_1_14`） |

每工况字段：

| 字段 | 形状 | 单位 / 说明 |
|------|------|-------------|
| `Time_s` | `(N_raw,)` | 绝对时间，**dt = 1 s** |
| `Temp_K` | `(N_raw,)` | 温度 [K] |
| `Conc` | `(N_raw,)` | 浓度（无量纲） |
| `L_mid_um` | `(1000,)` | 粒径中心 [μm]，约 0.5~999.5 |
| `psd` | `(N_raw, 1000)` | 数密度 n(L,t) [#/μm/L] |

检查工具：`scripts/00_check_mat.py`。

---

## 3. 预处理 → `dataset_3s.npz`

| 操作 | 说明 |
|------|------|
| 时间抽稀 | 每 3 个 1s 点取 1 个 → **dt = 3 s** |
| PSD | 与时间在相同索引上抽稀 |

```powershell
.\.venv\Scripts\python.exe scripts\01_preprocess.py
```

**`.npz` 键**：

| 键 | 形状 | 说明 |
|----|------|------|
| `T`, `C` | `(15, n_time)` | 抽稀后序列 |
| `psd` | `(15, n_time, 1000)` | PSD 时空场 |
| `L` | `(1000,)` | 粒径轴 [μm] |
| `t` | `(n_time,)` | 绝对时间 [s] |
| `dt` | 标量 | **3.0** |
| `case_names` | `(15,)` | 工况名 |

当前仓库实测：`n_time ≈ 2901`，覆盖 0~8700 s。  
**改 Branch 维数不需重跑预处理**（切窗时从 `T` 取输出段作 `T_future`）。

---

## 4. 训练样本（滑动窗口）

### 4.1 时间结构

```
|<-------- 输入 15 min (300@3s) -------->|<----- 输出 5 min (100@3s) ----->|
   T_hist, C_hist, PSD_hist n(L,t)         标签 PSD/C；Branch 末段 T_future
```

| 参数 | 秒 | 点数 | 配置 |
|------|-----|------|------|
| 输入窗 | 900 | 300 | `in_seconds` → `n_in` |
| 输出窗 | 300 | 100 | `out_seconds` → `n_out` |
| 滑动步长 | 300 | 100 点 | `window_stride_pts` |

起点 `t0` 每前进 100 点（5 min）切一个样本；每工况约 26 窗（`n_time=2901` 时）。

### 4.2 Branch 输入（39100 维）

归一化后拼接（`assemble_branch`）：

| 段 | 维数 | 源 | 归一化 |
|----|------|-----|--------|
| `T_hist` | 300 | 输入窗 T 全点 | min-max |
| `C_hist` | 300 | 输入窗 C 全点 | min-max |
| `PSD_hist` | 38400 | 输入窗 psd (300,1000)，**300 时刻×128 L** | ÷ `n_scale` |
| `T_future_plan` | 100 | 输出窗真值 T 全点 | min-max |

```
branch_dim = 300 + 300 + 300×128 + 100 = 39100
```

- 传感器索引：`arange(300)` / `arange(100)`；L 向 `linspace(0,999,128)`。  
- PSD 展平：**先时刻、后粒径**。

训练时 `T_future` = 仿真真值 `T[t0+300 : t0+400]`。  
推理/NMPC 时 `T_future` = 计划温度（每窗 100 点）。

### 4.3 Trunk（固定网格）

| 分支 | 形状 | 内容 |
|------|------|------|
| PSD | `(20000, 2)` | `[L_norm, τ_norm]`，200×100 |
| 浓度 | `(100, 1)` | 仅 `τ_norm`，与 PSD **共用** 100 个 τ |

τ：`0, 3, …, 297 s`；`τ_norm = τ / 300`。

### 4.4 训练标签

| 标签 | 形状 | 内容 |
|------|------|------|
| `label_psd` | `(N, 20000)` | 输出窗 PSD，L 评估 200 点，归一化 |
| `label_conc` | `(N, 100)` | 输出窗浓度，归一化 |

损失：`MSE(PSD) + λ_nonneg·非负 + λ_conc·MSE(浓度)`。

### 4.5 划分与归一化

- 工况划分：train 11 / val 3 / holdout 1（`split_seed=42`）。  
- `T_min/max`, `C_min/max`, `L_max`, `n_scale` **仅由训练工况**统计。

---

## 5. 训练产物 `results/<run>/`

| 文件 | 内容 |
|------|------|
| `config.json` | `branch_dim`, `window`, `model`, `train` |
| `norm_params.npz` | 归一化、传感器索引、`trunk_grid`, `trunk_conc_grid` |
| `weights/best.pt` | 最优权重（**Branch 输入维须为 39100**） |

校验：`python scripts/check_branch_dim.py`。

---

## 6. 推理

入口：`donpbe/predictor.py` → `Predictor`。

### 6.1 单窗输入

| 输入 | 形状 | 说明 |
|------|------|------|
| `T_seq` | `(300,)` | 输入窗温度 |
| `C_seq` | `(300,)` | 输入窗浓度 |
| `psd_seq` | `(300, 1000)` | 输入窗 **n(L,t)** 历史 |
| `T_future` | `(100,)` | **本窗** 5min 计划温度 |

组装 → **Branch (39100)** → 前向。

时间：`t_out[k] = t_in_end + (k+1)×3 s`。

### 6.2 单窗输出（反归一化）

| 输出 | 形状 | 说明 |
|------|------|------|
| `pred_psd` | `(100, 200)` 或 `(100, 1000)` | n(L,τ) |
| `pred_conc` | `(100,)` | C(τ) |

### 6.3 多窗自回归

NMPC 等场景：全长 `future_T` 长度 `n_windows×100`（默认 300）。  
第 `w` 窗：`T_future = future_T[w×100:(w+1)×100]`；预测 T/C/PSD 滚入下一窗输入（`roll_pack_forward`）。

| 拼接输出 | 形状 |
|----------|------|
| `pred_psd` | `(n_windows×100, n_L)` |
| `pred_conc` | `(n_windows×100,)` |

### 6.4 常用脚本

| 脚本 | 用途 |
|------|------|
| `02_train.py` | 训练 |
| `04_predict.py` | 单窗预测 |
| `05_rolling_predict.py` | 全程 teacher-forcing 滚动 |
| `07_autoregressive_rolling.py` | 自回归滚动 |
| `10_infer_preheat_planned_T.py` | 用 NMPC 快照 `buf_*` + `future_T` 离线推理 |

---

## 7. 维度速查

| 名称 | 值 |
|------|-----|
| `dt` | 3 s |
| `n_in` / `n_out` | 300 / 100 |
| `n_L` | 1000 |
| Branch L 采样 | 128 / 时刻 |
| Trunk L 评估 | 200 |
| `branch_dim` | **39100** |
| `n_query` | 20000 |

---

## 8. 与 NMPC 对齐要点

1. Plant 提供 **300@3s** 的 `T, C, psd(300,1000)`。  
2. 计划温度起点：**`buffer.T[-1]`**。  
3. NMPC 生成 `future_T(300,)`；DON **每窗**只用 **100 点**切片进 Branch。  
4. 部署：NMPC `import_don_assets.py` 导入 `norm_params.npz` + `best.pt`。  
5. 离线对照：用 `cache/first_opt_snapshot.npz` 的 `buf_*`，勿用与快照时刻不一致的预热缓存。

---

## 9. 源码索引

| 模块 | 文件 |
|------|------|
| 配置 | `donpbe/config.py` |
| 预处理 | `donpbe/preprocess.py`, `scripts/01_preprocess.py` |
| 数据集 | `donpbe/dataset.py` |
| 网络 | `donpbe/model.py` |
| 推理 | `donpbe/predictor.py` |
| 训练 | `donpbe/trainer.py`, `scripts/02_train.py` |
