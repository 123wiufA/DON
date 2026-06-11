# DON DeepONet：Branch / Trunk / 输出规格

与 [`DATA_IO.md`](DATA_IO.md) 配套：本文只描述 **送入网络的张量** 与 **模型输出**，不含原始 `.mat` 流水线。

代码：`donpbe/dataset.py`（`assemble_branch`）、`donpbe/model.py`、`donpbe/config.py`。

---

## 1. 单窗算子映射

```
输入 u（15min）                    DeepONet                    输出（5min）
─────────────────────────────────────────────────────────────────────────
T, C, PSD_hist n(L,t)     +     Trunk [L,τ] / [τ]     →     n(L,τ), C(τ)
T_future_plan (100)
```

| 时间轴 | 点数 @ 3s | 时长 |
|--------|-----------|------|
| 输入窗 | 300 | 15 min |
| 输出窗 τ | 100 | 5 min |

输出相对时间：`τ[k] = k×3 s`，`τ_norm = τ/300`。  
输出绝对时间：`t_out[k] = t_in_end + (k+1)×3 s`。

---

## 2. 网络结构

```
Branch (39100) ──► BranchNet ──► b (128)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
         Trunk_PSD (20000×2)              Trunk_Conc (100×1)
                    │                               │
                    ▼                               ▼
         psd = b·Φ_psd^T + bias            conc = b·Φ_conc^T + bias_c
```

| 部件 | PSD | 浓度 |
|------|-----|------|
| BranchNet | 共用 | 共用 |
| TrunkNet | 独立（2 维输入） | 独立（1 维输入） |
| 查询 τ | **相同 100 点** | **相同 100 点** |

浓度为算子输出 `b·Φ^T`，非 Branch 后接 MLP。

---

## 3. Branch（39100 维，float32，已归一化）

### 3.1 拼接顺序

```
[ T_hist(300) | C_hist(300) | PSD_hist(300×128) | T_future_plan(100) ]
```

```
branch_dim = n_in + n_in + n_in×n_L_sensors + n_out = 39100
```

### 3.2 索引布局

| 段 | 索引区间 | 维数 |
|----|----------|------|
| `T_hist` | [0, 300) | 300 |
| `C_hist` | [300, 600) | 300 |
| `PSD_hist` | [600, 39000) | 38400 |
| `T_future_plan` | [39000, 39100) | 100 |

### 3.3 各段说明

**T_hist / C_hist**

- 源：输入窗 300 点 @ 3s，**无下采样**  
- 归一化：`(x - min)/(max - min)`

**PSD_hist n(L,t)**

- 源：`psd_in[t,L]`，形状 `(300, 1000)`  
- 每个时刻 `t` 在 L 轴取 **128** 点（`L_sensor_idx` 均匀）  
- 归一化：`n / n_scale`  
- 展平：先时刻 `t=0…299`，每块 128 维

**T_future_plan**

- 源：本窗 **100** 点计划/真值温度 @ 3s  
- 训练：`T[输出段]`  
- NMPC：`future_T[w×100:(w+1)×100]`（全长 300 点计划仅用于切片）

### 3.4 批形状

| 场景 | 形状 |
|------|------|
| 单样本 | `(39100,)` |
| 训练批 | `(B, 39100)` |
| 潜向量 | `(B, 128)` |

---

## 4. Trunk（全样本固定）

存于 `norm_params.npz`：`trunk_grid`, `trunk_conc_grid`。

### 4.1 Trunk_PSD

| 项 | 值 |
|----|-----|
| 形状 | `(20000, 2)` |
| 列 0 | `L_norm = L / L_max`（200 评估点） |
| 列 1 | `τ_norm`（100 点，每 L 重复） |
| 展平 | `idx = l_idx × 100 + τ_idx` |

### 4.2 Trunk_Conc

| 项 | 值 |
|----|-----|
| 形状 | `(100, 1)` |
| 内容 | 与 Trunk_PSD 相同的 `τ_norm` |

### 4.3 全粒度推理（可选）

| 项 | 评估网格 | 全网格 |
|----|----------|--------|
| Trunk | 200×100 | 1000×100 |
| `pred_psd` | `(100, 200)` | `(100, 1000)` |

---

## 5. 模型输出

反归一化后：

| 输出 | 归一化形状 | reshape | 物理量 | 单位 |
|------|------------|---------|--------|------|
| PSD | `(20000,)` | `(100, 200)` | n(L,τ) | #/μm/L |
| 浓度 | `(100,)` | `(100,)` | C(τ) | 无量纲 |

```
n = max(psd_norm × n_scale, 0)
C = conc_norm × (C_max - C_min) + C_min
```

行索引 k ↔ 同一 τ：`pred_psd[k,:]` 与 `pred_conc[k]` 同时刻。

---

## 6. NMPC 15min 预测与 DON 5min 能力

| 层级 | 点数 @ 3s | 负责方 |
|------|-----------|--------|
| NMPC 计划温度（全长） | 300 | `plan_temperature` |
| DON 单窗 Branch 末段 | 100 | 当前窗切片 |
| DON 单窗输出 | 100 | Trunk + 内积 |
| 自回归窗数 | 3 | `pred_horizon / pred_seconds` |

DON **不**一次输出 15min；NMPC 用 3 次 5min 自回归拼接为 15min 预测。

---

## 7. 配置与资产

| 参数 | 默认值 |
|------|--------|
| `n_T_sensors` / `n_C_sensors` | 300 |
| `n_L_sensors` | 128 |
| `n_T_future_sensors` | 100 |
| `n_L_eval` | 200 |
| `branch_dim` | 39100 |
| `latent_dim` | 128 |

`norm_params.npz` 关键字段：`T/C_min/max`, `L_max`, `n_scale`, `*_sensor_idx`, `trunk_*`, `dt`, `n_in`, `n_out`。

---

## 8. 版本

| branch_dim | 状态 |
|------------|------|
| 278 | 废弃（末时刻 PSD） |
| 6550 | 废弃（50 点下采样） |
| **39100** | **当前（全分辨率）** |

更换 Branch 结构后须重训并更新 NMPC `assets/don_run`。

---

## 9. 相关

| 文档 / 代码 | 说明 |
|-------------|------|
| `docs/DATA_IO.md` | 原始数据 → 训练 → 推理 |
| `NMPC_Modular/docs/DATA_IO.md` | Plant → NMPC → DON |
| `NMPC_Modular/nmpckit/dataio/don_input.py` | 控制侧 `assemble_branch` |
