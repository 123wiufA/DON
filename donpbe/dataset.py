"""
数据集模块：把抽稀后的 .npz 切成「前 15min → 后 5min」的算子学习样本（多任务）。

算子映射设计
------------
输入函数 u（前 15min 的过程信息）── DeepONet ──► 后 5min 的 PSD n(L, τ) 与浓度 C(τ)

  Branch 输入（每个窗口一个向量，全分辨率无下采样）:
      [ T_hist(300) | C_hist(300) | PSD_hist(300×128) | T_future_plan(100) ]
      branch_dim = 39100
  Trunk 输入（查询坐标，所有窗口共享同一网格）:
      [ L_norm, τ_norm ]，τ 为预测窗口内的相对时间 (0 ~ 300s)
  标签:
      label_psd  : n(L, t_in + τ)，后 5min 的 PSD（展平为 n_query）
      label_conc : C(t_in + τ)，后 5min 的浓度序列（n_out 维）

窗口切分：在每个工况上以 ``window_stride_pts``（默认 100 点=5min）为步长滚动，
即 0–15→15–20、5–20→20–25 …… 每工况切出多个样本。

由于所有窗口共享同一 (L, τ) 评估网格，PSD 可用一次矩阵乘
``pred = branch_out @ trunk_out.T`` 高效预测整段。

划分：工况随机分为 train / val(测试集) / holdout(推理验证，完全不接触)，
归一化参数仅由训练工况统计，避免信息泄漏。
"""

import os
from typing import Dict, List, Tuple

import numpy as np


class Normalizer:
    """min-max / scale 归一化参数容器，仅由训练集统计得到。"""

    def __init__(self, T_min, T_max, C_min, C_max, L_max, n_scale,
                 out_seconds):
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self.C_min = float(C_min)
        self.C_max = float(C_max)
        self.L_max = float(L_max)
        self.n_scale = float(n_scale)
        self.out_seconds = float(out_seconds)

    def norm_T(self, x):
        return (x - self.T_min) / (self.T_max - self.T_min + 1e-12)

    def norm_C(self, x):
        return (x - self.C_min) / (self.C_max - self.C_min + 1e-12)

    def denorm_C(self, x):
        return x * (self.C_max - self.C_min) + self.C_min

    def norm_L(self, x):
        return x / self.L_max

    def norm_n(self, x):
        return x / self.n_scale

    def denorm_n(self, x):
        return x * self.n_scale

    def to_dict(self) -> Dict:
        return dict(T_min=self.T_min, T_max=self.T_max,
                    C_min=self.C_min, C_max=self.C_max,
                    L_max=self.L_max, n_scale=self.n_scale,
                    out_seconds=self.out_seconds)

    @classmethod
    def from_dict(cls, d: Dict) -> "Normalizer":
        return cls(d["T_min"], d["T_max"], d["C_min"], d["C_max"],
                   d["L_max"], d["n_scale"], d["out_seconds"])


def encode_psd_history(normalizer: Normalizer,
                       psd_in: np.ndarray,
                       T_sensor_idx: np.ndarray,
                       L_sensor_idx: np.ndarray) -> np.ndarray:
    """输入窗 PSD 历史 n(L,t)：与 T/C 共用时间采样索引，每时刻在 L 轴采 n_L_sensors 点。

    Parameters
    ----------
    psd_in : (n_in, n_L)  输入段完整时空场
    展平顺序：先时刻（``T_sensor_idx`` 顺序），每时刻一块 ``L_sensor_idx``。
    """
    psd_in = np.asarray(psd_in, dtype=np.float64)
    if psd_in.ndim != 2:
        raise ValueError(f"psd_in 须为 (n_in, n_L)，当前 shape={psd_in.shape}")
    parts = [
        normalizer.norm_n(psd_in[int(ti), L_sensor_idx])
        for ti in T_sensor_idx
    ]
    return np.concatenate(parts)


def assemble_branch(normalizer: Normalizer,
                    T_in: np.ndarray,
                    C_in: np.ndarray,
                    psd_in: np.ndarray,
                    T_future: np.ndarray,
                    T_sensor_idx: np.ndarray,
                    C_sensor_idx: np.ndarray,
                    L_sensor_idx: np.ndarray,
                    T_future_sensor_idx: np.ndarray) -> np.ndarray:
    """拼接 Branch 向量：历史 T/C + PSD 历史 n(L,t) + 输出窗计划温度。"""
    T_future = np.asarray(T_future, dtype=np.float64)
    need = int(T_future_sensor_idx.max()) + 1
    if T_future.size < need:
        raise ValueError(
            f"T_future 至少 {need} 点（输出窗），当前 {T_future.size}。")
    branch = np.concatenate([
        normalizer.norm_T(T_in[T_sensor_idx]),
        normalizer.norm_C(C_in[C_sensor_idx]),
        encode_psd_history(normalizer, psd_in, T_sensor_idx, L_sensor_idx),
        normalizer.norm_T(T_future[T_future_sensor_idx]),
    ])
    return branch.astype(np.float32)


class PBEWindowData:
    """从 .npz 构建滑动窗口算子样本，并按工况划分数据集。

    Parameters
    ----------
    npz_path : str
        预处理生成的 .npz 路径。
    cfg : Config
        全局配置（见 config.py）。
    """

    def __init__(self, npz_path: str, cfg):
        self.cfg = cfg
        self.win = cfg.window

        data = np.load(npz_path, allow_pickle=True)
        self.T = data["T"]              # (n_cases, n_time)
        self.C = data["C"]              # (n_cases, n_time)
        self.psd = data["psd"]          # (n_cases, n_time, n_L)
        self.L = data["L"]              # (n_L,)
        self.t = data["t"]              # (n_time,)
        self.case_names = [str(x) for x in data["case_names"]]
        self.dt = float(data["dt"])

        self.n_cases, self.n_time = self.T.shape
        self.n_L = self.L.size

        # 粒径采样索引
        self.L_sensor_idx = np.linspace(
            0, self.n_L - 1, self.win.n_L_sensors, dtype=int)
        self.L_eval_idx = np.linspace(
            0, self.n_L - 1, self.win.n_L_eval, dtype=int)
        self.L_eval = self.L[self.L_eval_idx]

        # 输入段温度/浓度：全分辨率 0..n_in-1（与 n_in 对齐，无下采样）
        self.T_sensor_idx = np.arange(self.win.n_in, dtype=int)
        self.C_sensor_idx = np.arange(self.win.n_in, dtype=int)
        # 输出段计划温度：全分辨率 0..n_out-1
        self.T_future_sensor_idx = np.arange(self.win.n_out, dtype=int)

        # 工况划分：train / val(测试集) / holdout(推理验证)
        self.train_cases, self.val_cases, self.holdout_cases = self._split_cases()

        # 归一化（仅用训练工况，避免信息泄漏）
        self.normalizer = self._fit_normalizer(self.train_cases)

        # 固定 Trunk 网格（所有窗口共享）
        self.trunk_grid = self._build_trunk_grid()       # (n_query, 2) PSD: [L, τ]
        self.trunk_conc_grid = self._build_trunk_conc_grid()  # (n_out, 1) 浓度: [τ]

    # ------------------------------------------------------------------
    # 划分与归一化
    # ------------------------------------------------------------------
    def _case_index(self, name: str) -> int:
        return self.case_names.index(name)

    def _split_cases(self) -> Tuple[List[int], List[int], List[int]]:
        """随机把工况划分为 训练/测试(val)/推理验证(holdout)。

        若 config 中显式给出 train_cases/val_cases/holdout_cases，则优先使用；
        否则按 n_train_cases / n_val_cases / n_holdout_cases 用 split_seed 随机划分。
        """
        tc = self.cfg.train

        # 显式指定优先
        if tc.train_cases and tc.val_cases and tc.holdout_cases:
            train = [self._case_index(n) for n in tc.train_cases if n in self.case_names]
            val = [self._case_index(n) for n in tc.val_cases if n in self.case_names]
            holdout = [self._case_index(n) for n in tc.holdout_cases if n in self.case_names]
            return train, val, holdout

        # 随机划分
        rng = np.random.default_rng(tc.split_seed)
        perm = rng.permutation(self.n_cases).tolist()
        n_tr, n_va, n_ho = tc.n_train_cases, tc.n_val_cases, tc.n_holdout_cases
        if n_tr + n_va + n_ho > self.n_cases:
            raise ValueError(
                f"划分数量之和 {n_tr+n_va+n_ho} 超过工况总数 {self.n_cases}。")
        train = sorted(perm[:n_tr])
        val = sorted(perm[n_tr:n_tr + n_va])
        holdout = sorted(perm[n_tr + n_va:n_tr + n_va + n_ho])
        if not train:
            raise ValueError("训练工况为空，请检查划分配置。")
        return train, val, holdout

    def _fit_normalizer(self, train_idx: List[int]) -> Normalizer:
        T_tr = self.T[train_idx]
        C_tr = self.C[train_idx]
        psd_tr = self.psd[train_idx]
        pos = psd_tr[psd_tr > 0]
        n_scale = float(np.percentile(pos, 99)) if pos.size else 1.0
        norm = Normalizer(
            T_min=T_tr.min(), T_max=T_tr.max(),
            C_min=C_tr.min(), C_max=C_tr.max(),
            L_max=float(self.L.max()), n_scale=n_scale,
            out_seconds=self.win.out_seconds,
        )
        print(f"[Data] 归一化: T=[{norm.T_min:.2f},{norm.T_max:.2f}], "
              f"C=[{norm.C_min:.4f},{norm.C_max:.4f}], "
              f"L_max={norm.L_max:.1f}, n_scale={norm.n_scale:.4e}")
        return norm

    def _build_trunk_grid(self) -> np.ndarray:
        """构建 (n_query, 2) 的 [L_norm, τ_norm] 网格，L 外层、τ 内层。"""
        L_norm = self.normalizer.norm_L(self.L_eval)            # (n_L_eval,)
        tau = np.arange(self.win.n_out, dtype=np.float64) * self.dt
        tau_norm = tau / self.win.out_seconds                   # (n_out,)
        # 顺序：index = l * n_out + tau
        Lg, Tg = np.meshgrid(L_norm, tau_norm, indexing="ij")   # (n_L_eval, n_out)
        grid = np.stack([Lg.ravel(), Tg.ravel()], axis=-1)
        return grid.astype(np.float32)

    def _build_trunk_conc_grid(self) -> np.ndarray:
        """构建浓度算子 Trunk 网格 (n_out, 1)：仅输出窗口内归一化时间 τ。"""
        tau = np.arange(self.win.n_out, dtype=np.float64) * self.dt
        tau_norm = tau / self.win.out_seconds
        return tau_norm.reshape(-1, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # 滑动窗口样本构建
    # ------------------------------------------------------------------
    def _window_starts(self) -> List[int]:
        """返回所有合法窗口起点（抽稀后点索引）。"""
        last_start = self.n_time - self.win.n_window
        if last_start < 0:
            raise ValueError(
                f"单工况时间点 {self.n_time} 小于窗口长度 {self.win.n_window}，"
                "请减小 in_seconds/out_seconds 或 raw_stride。")
        return list(range(0, last_start + 1, self.win.window_stride_pts))

    def _build_branch(self, ci: int, t0: int) -> np.ndarray:
        """构建单个窗口的 Branch 向量。"""
        out_start = t0 + self.win.n_in
        return assemble_branch(
            self.normalizer,
            self.T[ci, t0:t0 + self.win.n_in],
            self.C[ci, t0:t0 + self.win.n_in],
            self.psd[ci, t0:t0 + self.win.n_in],
            self.T[ci, out_start:out_start + self.win.n_out],
            self.T_sensor_idx, self.C_sensor_idx, self.L_sensor_idx,
            self.T_future_sensor_idx)

    def _build_label_psd(self, ci: int, t0: int) -> np.ndarray:
        """构建单个窗口的 PSD 标签向量 (n_query,)，顺序与 trunk_grid 一致。"""
        out_start = t0 + self.win.n_in
        out_slice = self.psd[ci, out_start:out_start + self.win.n_out]  # (n_out, n_L)
        out_eval = out_slice[:, self.L_eval_idx]                        # (n_out, n_L_eval)
        # 转成 (n_L_eval, n_out) 后展平：index = l*n_out + tau
        label = out_eval.T.ravel()
        return self.normalizer.norm_n(label).astype(np.float32)

    def _build_label_conc(self, ci: int, t0: int) -> np.ndarray:
        """构建单个窗口的浓度标签向量 (n_out,)：后 5min 的归一化浓度序列。"""
        out_start = t0 + self.win.n_in
        c_out = self.C[ci, out_start:out_start + self.win.n_out]        # (n_out,)
        return self.normalizer.norm_C(c_out).astype(np.float32)

    def build_split(self, which: str
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建某个划分的 (branch, label_psd, label_conc) 数组。

        Returns
        -------
        branch     : (N, branch_dim) float32
        label_psd  : (N, n_query)    float32   后5min PSD（归一化）
        label_conc : (N, n_out)      float32   后5min 浓度（归一化）
        """
        idx_map = {"train": self.train_cases,
                   "val": self.val_cases,
                   "holdout": self.holdout_cases}
        cases = idx_map[which]
        starts = self._window_starts()

        branches, psds, concs = [], [], []
        for ci in cases:
            for t0 in starts:
                branches.append(self._build_branch(ci, t0))
                psds.append(self._build_label_psd(ci, t0))
                concs.append(self._build_label_conc(ci, t0))

        if not branches:
            bd = self.cfg.branch_dim
            return (np.zeros((0, bd), np.float32),
                    np.zeros((0, self.cfg.n_query), np.float32),
                    np.zeros((0, self.win.n_out), np.float32))
        return np.stack(branches), np.stack(psds), np.stack(concs)

    # ------------------------------------------------------------------
    # 信息打印
    # ------------------------------------------------------------------
    def summary(self) -> None:
        starts = self._window_starts()
        names = lambda idx: [self.case_names[i] for i in idx]
        print("=" * 60)
        print(f"[Data] 工况总数={self.n_cases}, 抽稀后时间点={self.n_time}, "
              f"dt={self.dt}s, L={self.n_L}")
        print(f"[Data] 窗口: 输入 {self.win.n_in} 点(15min) + "
              f"输出 {self.win.n_out} 点(5min), 步长 {self.win.window_stride_pts} 点")
        n_win = len(starts)
        n_tr, n_va, n_ho = (len(self.train_cases), len(self.val_cases),
                            len(self.holdout_cases))
        print(f"[Data] 每工况切片数={n_win} (滚动步长 "
              f"{self.win.window_stride_pts}点={self.win.window_stride_pts*self.dt:.0f}s)")
        print(f"[Data] 训练集 {n_tr} 工况: {names(self.train_cases)}")
        print(f"       → 训练切片总数 = {n_tr} × {n_win} = {n_tr*n_win}")
        print(f"[Data] 测试集/监控 {n_va} 工况: {names(self.val_cases)}")
        print(f"       → 测试切片总数 = {n_va} × {n_win} = {n_va*n_win}")
        print(f"[Data] 推理验证holdout {n_ho} 工况(不接触): {names(self.holdout_cases)}")
        print(f"       → holdout 切片总数 = {n_ho} × {n_win} = {n_ho*n_win}")
        print(f"[Data] 全部切片合计 = {(n_tr+n_va+n_ho)*n_win}")
        print(f"[Data] branch_dim={self.cfg.branch_dim}, "
              f"n_query(PSD)={self.cfg.n_query}, n_out(浓度)={self.win.n_out}")
        print("=" * 60)

    def save_norm_params(self, path: str) -> None:
        """保存归一化与网格参数，供预测脚本复用。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            trunk_grid=self.trunk_grid,
            trunk_conc_grid=self.trunk_conc_grid,
            L_eval=self.L_eval,
            L_eval_idx=self.L_eval_idx,
            L_sensor_idx=self.L_sensor_idx,
            T_sensor_idx=self.T_sensor_idx,
            C_sensor_idx=self.C_sensor_idx,
            T_future_sensor_idx=self.T_future_sensor_idx,
            dt=self.dt,
            **self.normalizer.to_dict(),
        )
