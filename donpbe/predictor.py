"""
预测器模块：加载训练好的模型，对指定工况做单窗口或全程滚动预测。

供脚本 04_predict.py（自定义起始时刻的单窗口预测）与
05_rolling_predict.py（全程滚动预测）复用，避免逻辑重复。
"""

import os
from typing import Optional, Dict

import numpy as np
import torch

from .config import apply_run_config
from .dataset import PBEWindowData
from .model import DeepONet


class Predictor:
    """封装「加载模型 + 构建输入 + 预测 + 反归一化」的推理流程。"""

    def __init__(self, run_dir: str, npz_path: str, cfg, device):
        self.cfg = apply_run_config(cfg, run_dir)
        self.win = self.cfg.window
        self.device = device

        # 复用数据管线（提供归一化、网格、原始数组与划分信息）
        self.data = PBEWindowData(npz_path, self.cfg)
        self.nrm = self.data.normalizer
        self.dt = self.data.dt
        self.n_out = self.win.n_out
        self.n_L_eval = self.win.n_L_eval
        self.L_eval = self.data.L_eval
        self.tau_seconds = np.arange(self.n_out) * self.dt

        # 全粒度网格：利用 DeepONet 的 Trunk 可查询任意粒径坐标的特性，
        # 在完整 L 网格（n_L 个点）上预测，得到全分辨率时空场用于可视化。
        self.L_full = self.data.L                       # (n_L,)
        self.n_L_full = self.L_full.size
        self.trunk_full = self._build_trunk_full()      # (n_L_full * n_out, 2)

        # 构建并加载模型
        self.model = DeepONet(
            branch_dim=self.cfg.branch_dim, trunk_dim=2,
            branch_hiddens=self.cfg.model.branch_hiddens,
            trunk_hiddens=self.cfg.model.trunk_hiddens,
            latent_dim=self.cfg.model.latent_dim,
            activation=self.cfg.model.activation,
            n_out=self.n_out,
            conc_trunk_hiddens=self.cfg.model.conc_trunk_hiddens).to(device)
        weights = os.path.join(run_dir, "weights", "best.pt")
        self.model.load_state_dict(
            torch.load(weights, map_location=device, weights_only=True))
        self.model.eval()
        self.trunk = torch.from_numpy(self.data.trunk_grid).to(device)
        self.trunk_conc = torch.from_numpy(self.data.trunk_conc_grid).to(device)
        self.trunk_full_t = torch.from_numpy(self.trunk_full).to(device)
        self.run_dir = run_dir

    # ------------------------------------------------------------------
    def _build_trunk_full(self) -> np.ndarray:
        """构建全粒度 Trunk 网格 (n_L_full * n_out, 2)，顺序 index = l*n_out + tau。"""
        L_norm = self.nrm.norm_L(self.L_full)
        tau = np.arange(self.n_out, dtype=np.float64) * self.dt
        tau_norm = tau / self.win.out_seconds
        Lg, Tg = np.meshgrid(L_norm, tau_norm, indexing="ij")   # (n_L_full, n_out)
        grid = np.stack([Lg.ravel(), Tg.ravel()], axis=-1)
        return grid.astype(np.float32)

    def _build_branch_from_seq(self, T_seq: np.ndarray, C_seq: np.ndarray,
                               psd_seq: np.ndarray,
                               T_future: np.ndarray) -> np.ndarray:
        """由输入窗 T/C/PSD 序列 + 输出窗计划 T 构建 Branch。"""
        n_in, n_out = self.win.n_in, self.win.n_out
        if T_seq.size != n_in or C_seq.size != n_in:
            raise ValueError(f"T/C 序列长度须为 {n_in}。")
        psd_seq = np.asarray(psd_seq, dtype=np.float64)
        if psd_seq.shape[0] != n_in:
            raise ValueError(
                f"psd_seq 须 ({n_in}, n_L)，当前 {psd_seq.shape}。")
        T_future = np.asarray(T_future, dtype=np.float64)
        if T_future.size != n_out:
            raise ValueError(f"T_future 须 {n_out} 点，当前 {T_future.size}。")
        from .dataset import assemble_branch
        return assemble_branch(
            self.nrm, T_seq, C_seq, psd_seq, T_future,
            self.data.T_sensor_idx, self.data.C_sensor_idx,
            self.data.L_sensor_idx, self.data.T_future_sensor_idx)

    @staticmethod
    def _hold_T_future(T_seq: np.ndarray, n_out: int) -> np.ndarray:
        return np.full(n_out, T_seq[-1], dtype=np.float64)

    @torch.no_grad()
    def _forward_branch(self, branch: np.ndarray, full_resolution: bool = True):
        """由 Branch 向量前向，返回真实尺度 (psd, conc)。"""
        trunk = self.trunk_full_t if full_resolution else self.trunk
        n_L = self.n_L_full if full_resolution else self.n_L_eval
        psd, conc = self.model(
            torch.from_numpy(branch[None, :]).to(self.device),
            trunk, self.trunk_conc)
        psd = psd.cpu().numpy()[0]
        conc = conc.cpu().numpy()[0]
        pred_psd = self.nrm.denorm_n(np.maximum(psd, 0)).reshape(
            n_L, self.n_out).T
        pred_conc = self.nrm.denorm_C(conc)
        return pred_psd, pred_conc

    @torch.no_grad()
    def _forward_full(self, ci: int, t0: int):
        """全粒度前向：返回真实尺度 (psd_grid (n_out, n_L_full), conc (n_out,))。"""
        branch = self.data._build_branch(ci, t0)[None, :]
        psd, conc = self.model(
            torch.from_numpy(branch).to(self.device),
            self.trunk_full_t, self.trunk_conc)
        psd = psd.cpu().numpy()[0]
        conc = conc.cpu().numpy()[0]
        pred_psd = self.nrm.denorm_n(np.maximum(psd, 0)).reshape(
            self.n_L_full, self.n_out).T          # (n_out, n_L_full)
        pred_conc = self.nrm.denorm_C(conc)
        return pred_psd, pred_conc

    # ------------------------------------------------------------------
    def case_index(self, case_name: str) -> int:
        if case_name not in self.data.case_names:
            raise ValueError(
                f"工况 {case_name!r} 不在数据中。可选: {self.data.case_names}")
        return self.data.case_names.index(case_name)

    @torch.no_grad()
    def _forward(self, ci: int, t0: int):
        """对单个窗口前向，返回真实尺度的 (psd_grid, conc) 预测。"""
        branch = self.data._build_branch(ci, t0)[None, :]
        psd, conc = self.model(
            torch.from_numpy(branch).to(self.device), self.trunk, self.trunk_conc)
        psd = psd.cpu().numpy()[0]
        conc = conc.cpu().numpy()[0]
        pred_psd = self.nrm.denorm_n(np.maximum(psd, 0)).reshape(
            self.n_L_eval, self.n_out).T          # (n_out, n_L_eval)
        pred_conc = self.nrm.denorm_C(conc)        # (n_out,)
        return pred_psd, pred_conc

    def predict_window(self, case_name: str, start_seconds: float) -> Dict:
        """从 ``start_seconds`` 开始取前 15min 作为输入，预测随后的 5min。

        Returns 含 pred/true 的 PSD 网格、浓度序列、各类时间轴与误差。
        若该窗口的输出段超出数据范围，则 true_* 为 None（仅给预测）。
        """
        ci = self.case_index(case_name)
        t0 = int(round(start_seconds / self.dt))
        if t0 < 0 or t0 + self.win.n_in > self.data.n_time:
            raise ValueError(
                f"起始时刻 {start_seconds}s 不合法：输入段需落在 "
                f"[0, {(self.data.n_time - self.win.n_in) * self.dt:.0f}]s 内。")

        pred_psd, pred_conc = self._forward(ci, t0)

        out_start = t0 + self.win.n_in
        out_end = out_start + self.n_out
        t_out_abs = (out_start + np.arange(self.n_out)) * self.dt

        has_truth = out_end <= self.data.n_time
        if has_truth:
            true_psd = self.nrm.denorm_n(
                self.data._build_label_psd(ci, t0)).reshape(
                self.n_L_eval, self.n_out).T
            true_conc = self.nrm.denorm_C(self.data._build_label_conc(ci, t0))
        else:
            true_psd = true_conc = None

        return dict(
            case=case_name,
            input_range=(t0 * self.dt, (t0 + self.win.n_in) * self.dt),
            output_range=(out_start * self.dt, out_end * self.dt),
            tau_seconds=self.tau_seconds,
            t_out_abs=t_out_abs,
            L_eval=self.L_eval,
            pred_psd=pred_psd, true_psd=true_psd,
            pred_conc=pred_conc, true_conc=true_conc,
        )

    def rolling_predict(self, case_name: str,
                        stride_seconds: Optional[float] = None,
                        full_resolution: bool = True) -> Dict:
        """全程滚动预测：在整条工况上滚动，拼接出从 15min 到结尾的预测轨迹。

        默认步长 = 输出窗口长度(5min)，使各窗口输出段首尾相接、无缝覆盖。
        每个窗口都以「真实的前 15min」为输入（teacher forcing，非自回归）。

        full_resolution=True 时在完整粒径网格 (n_L) 上预测，返回全粒度时空场；
        此时 true_psd 取原始仿真 PSD（真实尺度，无下采样）。
        """
        ci = self.case_index(case_name)
        stride_pts = (int(round(stride_seconds / self.dt))
                      if stride_seconds else self.n_out)

        last_start = self.data.n_time - self.win.n_window
        starts = list(range(0, last_start + 1, stride_pts))

        L_axis = self.L_full if full_resolution else self.L_eval

        # 第一个窗口的输入段(0~15min)作为上下文：取真实仿真值
        n_in = self.win.n_in
        ctx_t = np.arange(n_in) * self.dt                       # 0 ~ (n_in-1)*dt
        ctx_conc = self.data.C[ci, :n_in].astype(np.float64)
        ctx_psd_raw = self.data.psd[ci, :n_in, :]               # (n_in, n_L)
        ctx_psd = (ctx_psd_raw if full_resolution
                   else ctx_psd_raw[:, self.data.L_eval_idx])

        t_list, psd_pred_list, psd_true_list = [], [], []
        conc_pred_list, conc_true_list = [], []
        for t0 in starts:
            out_start = t0 + self.win.n_in
            t_abs = (out_start + np.arange(self.n_out)) * self.dt
            if full_resolution:
                pred_psd, pred_conc = self._forward_full(ci, t0)
                # 真值直接取原始全粒度 PSD（真实尺度）
                true_psd = self.data.psd[ci, out_start:out_start + self.n_out, :]
            else:
                pred_psd, pred_conc = self._forward(ci, t0)
                true_psd = self.nrm.denorm_n(
                    self.data._build_label_psd(ci, t0)).reshape(
                    self.n_L_eval, self.n_out).T
            true_conc = self.nrm.denorm_C(self.data._build_label_conc(ci, t0))

            t_list.append(t_abs)
            psd_pred_list.append(pred_psd)
            psd_true_list.append(np.asarray(true_psd))
            conc_pred_list.append(pred_conc)
            conc_true_list.append(true_conc)

        return dict(
            case=case_name,
            n_windows=len(starts),
            full_resolution=full_resolution,
            split_seconds=float(n_in * self.dt),    # 输入/预测分界(15min)
            t_abs=np.concatenate(t_list),
            L_eval=L_axis,
            pred_psd=np.concatenate(psd_pred_list, axis=0),   # (T_total, n_L)
            true_psd=np.concatenate(psd_true_list, axis=0),
            pred_conc=np.concatenate(conc_pred_list),
            true_conc=np.concatenate(conc_true_list),
            # 输入段(0~15min)上下文（真实仿真值）
            ctx_t=ctx_t, ctx_psd=ctx_psd, ctx_conc=ctx_conc,
        )

    def rolling_predict_autoregressive(
            self, case_name: str,
            start_seconds: float = 0.0,
            n_windows: Optional[int] = None,
            T_source: str = "true",
            full_resolution: bool = True) -> Dict:
        """自回归滚动预测：每窗预测 5min 后，将预测 C/PSD 滚入下一窗输入。

        与 ``rolling_predict``（teacher forcing）不同，本方法第 2 窗起输入窗内的
        浓度与 PSD 来自模型预测，误差会累积。

        Parameters
        ----------
        start_seconds : float
            初始输入窗起点（取随后 15min 真实数据作为第 1 窗输入）。
        n_windows : int, optional
            滚动窗数；默认直到仿真数据结束。
        T_source : str
            更新输入窗时温度段的来源：
            - ``true``：仍用仿真真值温度（仅 C/PSD 自回归，便于与 05 对比）
            - ``hold``：保持末点温度不变
        full_resolution : bool
            是否在完整粒径网格上预测。
        """
        if T_source not in ("true", "hold"):
            raise ValueError("T_source 须为 'true' 或 'hold'。")

        ci = self.case_index(case_name)
        n_in, n_out = self.win.n_in, self.n_out
        t0 = int(round(start_seconds / self.dt))
        if t0 < 0 or t0 + n_in > self.data.n_time:
            raise ValueError(
                f"起始时刻 {start_seconds}s 不合法：需留出 {n_in} 点输入窗。")

        T_seq = self.data.T[ci, t0:t0 + n_in].astype(np.float64).copy()
        C_seq = self.data.C[ci, t0:t0 + n_in].astype(np.float64).copy()
        psd_seq = self.data.psd[ci, t0:t0 + n_in, :].astype(np.float64).copy()

        win_end = t0 + n_in
        max_win = (self.data.n_time - win_end) // n_out
        if max_win <= 0:
            raise ValueError("起始点之后不足以做 1 个 5min 预测窗。")
        n_win = min(n_windows, max_win) if n_windows else max_win

        L_axis = self.L_full if full_resolution else self.L_eval
        ctx_t = np.arange(n_in) * self.dt + t0 * self.dt
        ctx_psd = psd_seq.copy() if full_resolution else psd_seq[:, self.data.L_eval_idx]
        ctx_conc = C_seq.copy()

        t_list, psd_pred_list, psd_true_list = [], [], []
        conc_pred_list, conc_true_list = [], []

        for _ in range(n_win):
            if T_source == "true":
                T_future = self.data.T[ci, win_end:win_end + n_out].astype(np.float64)
            else:
                T_future = self._hold_T_future(T_seq, n_out)

            branch = self._build_branch_from_seq(
                T_seq, C_seq, psd_seq, T_future)
            pred_psd, pred_conc = self._forward_branch(branch, full_resolution)

            t_abs = (win_end + np.arange(n_out)) * self.dt
            if full_resolution:
                true_psd = self.data.psd[ci, win_end:win_end + n_out, :]
            else:
                true_psd = self.data.psd[ci, win_end:win_end + n_out, :][:, self.data.L_eval_idx]
            true_conc = self.data.C[ci, win_end:win_end + n_out].astype(np.float64)

            t_list.append(t_abs)
            psd_pred_list.append(pred_psd)
            psd_true_list.append(np.asarray(true_psd))
            conc_pred_list.append(pred_conc)
            conc_true_list.append(true_conc)

            T_new = T_future

            T_seq = np.concatenate([T_seq[n_out:], T_new])
            C_seq = np.concatenate([C_seq[n_out:], pred_conc])
            psd_seq = np.concatenate([psd_seq[n_out:], pred_psd], axis=0)
            win_end += n_out

        return dict(
            case=case_name,
            mode="autoregressive",
            T_source=T_source,
            start_seconds=float(t0 * self.dt),
            n_windows=n_win,
            full_resolution=full_resolution,
            split_seconds=float(n_in * self.dt),
            t_abs=np.concatenate(t_list),
            L_eval=L_axis,
            pred_psd=np.concatenate(psd_pred_list, axis=0),
            true_psd=np.concatenate(psd_true_list, axis=0),
            pred_conc=np.concatenate(conc_pred_list),
            true_conc=np.concatenate(conc_true_list),
            ctx_t=ctx_t, ctx_psd=ctx_psd, ctx_conc=ctx_conc,
        )

    def rolling_predict_autoregressive_buffer(
            self,
            T_seq: np.ndarray,
            C_seq: np.ndarray,
            psd_seq: np.ndarray,
            n_windows: int,
            future_T: Optional[np.ndarray] = None,
            t_input_end: float = 0.0,
            rc_kph: Optional[float] = None,
            full_resolution: bool = True) -> Dict:
        """自回归滚动：任意 15min 输入窗 + 可选未来温度轨迹（非数据集工况）。

        ``psd_seq`` 须已在 DON 全粒度粒径轴 ``L_full`` 上。
        ``future_T`` 长度须为 ``n_windows * n_out``；未给则温度保持输入窗末点。
        """
        n_in, n_out = self.win.n_in, self.win.n_out
        T_seq = np.asarray(T_seq, dtype=np.float64).copy()
        C_seq = np.asarray(C_seq, dtype=np.float64).copy()
        psd_seq = np.asarray(psd_seq, dtype=np.float64).copy()
        if T_seq.size != n_in or C_seq.size != n_in or psd_seq.shape[0] != n_in:
            raise ValueError(
                f"输入窗须 {n_in} 点；收到 T={T_seq.size} C={C_seq.size} "
                f"psd={psd_seq.shape[0]}。")
        if psd_seq.shape[1] != self.n_L_full:
            raise ValueError(
                f"PSD 粒径维须为 {self.n_L_full}（DON L 网格），"
                f"当前 {psd_seq.shape[1]}。")
        need_T = n_windows * n_out
        if future_T is not None and np.asarray(future_T).size != need_T:
            raise ValueError(
                f"future_T 须 {need_T} 点，当前 {np.asarray(future_T).size}。")

        L_axis = self.L_full if full_resolution else self.L_eval
        ctx_t = np.linspace(
            t_input_end - (n_in - 1) * self.dt, t_input_end, n_in)
        ctx_psd = psd_seq.copy() if full_resolution else \
            psd_seq[:, self.data.L_eval_idx]
        ctx_conc = C_seq.copy()

        t_cursor = float(t_input_end)
        t_list, psd_list, conc_list, T_plan_list = [], [], [], []

        for w in range(n_windows):
            if future_T is not None:
                i0 = w * n_out
                T_future = np.asarray(future_T[i0:i0 + n_out], dtype=np.float64)
            else:
                T_future = self._hold_T_future(T_seq, n_out)

            branch = self._build_branch_from_seq(
                T_seq, C_seq, psd_seq, T_future)
            pred_psd, pred_conc = self._forward_branch(branch, full_resolution)
            t_out = t_cursor + np.arange(1, n_out + 1) * self.dt
            t_list.append(t_out)
            psd_list.append(pred_psd)
            conc_list.append(pred_conc)

            T_new = T_future
            T_plan_list.append(T_new)

            T_seq = np.concatenate([T_seq[n_out:], T_new])
            C_seq = np.concatenate([C_seq[n_out:], pred_conc])
            psd_seq = np.concatenate([psd_seq[n_out:], pred_psd], axis=0)
            t_cursor = t_out[-1]

        T_plan = np.concatenate(T_plan_list) if T_plan_list else np.array([])
        return dict(
            mode="autoregressive_buffer",
            rc_kph=rc_kph,
            n_windows=n_windows,
            pred_minutes=n_windows * n_out * self.dt / 60.0,
            full_resolution=full_resolution,
            split_seconds=float(n_in * self.dt),
            t_input_end=float(t_input_end),
            ctx_t=ctx_t, ctx_psd=ctx_psd, ctx_conc=ctx_conc,
            t_abs=np.concatenate(t_list),
            T_plan=T_plan,
            L_eval=L_axis,
            pred_psd=np.concatenate(psd_list, axis=0),
            pred_conc=np.concatenate(conc_list),
        )
