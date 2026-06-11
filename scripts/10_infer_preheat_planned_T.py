"""
脚本 10：35min 预热末 15min + NMPC 计划温度 → DON Branch 推理。

数据来源（默认与首次 NMPC 优化完全一致）：
  - ``first_opt_snapshot.npz`` 的 ``buf_*``：优化时刻的 Plant 输入窗（15min）
  - 同快照的 ``future_T``：最优 ``u_seq`` 规划的计划温度
  - 可选 ``--preheat_cache``：从预热缓存取末 15min（须与快照 ``t_window_end_s`` 一致）

流程：
  1. 组装 Branch（历史 T/C + PSD 历史 n(L,t) + 各窗 T_future）
  2. 自回归滚动预测（默认 3×5min = 15min，与 NMPC 预测时域一致）
  3. 若快照含 NMPC 预测 / Branch，做一致性对比
  4. 保存 npz 与图

用法::
    python scripts/10_infer_preheat_planned_T.py
    python scripts/10_infer_preheat_planned_T.py --run results/20260611_134539
    python scripts/10_infer_preheat_planned_T.py --run results/20260611_134539/weights/best.pt
    python scripts/10_infer_preheat_planned_T.py --n_windows 6 --pred_min 30
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                          "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from donpbe.config import get_default_config
from donpbe.device import setup_device
from donpbe.predictor import Predictor
from donpbe.utils import resolve_run_dir


def load_input_from_snapshot(snap_path: str) -> dict:
    """从首次优化快照 ``buf_*`` 还原输入窗（与 NMPC 优化时一致）。"""
    d = np.load(snap_path, allow_pickle=False)
    for key in ("buf_t", "buf_T", "buf_C", "buf_psd", "buf_L_um"):
        if key not in d.files:
            raise KeyError(f"快照缺少 {key}")
    t = d["buf_t"].astype(np.float64)
    return dict(
        t=t,
        T=d["buf_T"].astype(np.float64),
        C=d["buf_C"].astype(np.float64),
        psd_plant=d["buf_psd"].astype(np.float64),
        L_plant_um=d["buf_L_um"].astype(np.float64),
        t_end=float(t[-1]),
        warmup_seconds=float(t[-1]),
        source="snapshot",
    )


def load_preheat_input(cache_path: str, n_in: int, dt: float,
                       t_end_expected: float = None) -> dict:
    """预热缓存 → 最后 15min 输入窗。"""
    d = np.load(cache_path, allow_pickle=False)
    if d["hist_t"].size < n_in:
        raise ValueError(
            f"缓存采样 {d['hist_t'].size} 点 < 输入窗 {n_in} 点。")
    t = d["hist_t"][-n_in:].astype(np.float64)
    if n_in > 1 and not np.allclose(np.diff(t), dt, atol=1e-2):
        raise ValueError(f"缓存时间轴须均匀 {dt}s。")
    out = dict(
        t=t,
        T=d["hist_T"][-n_in:].astype(np.float64),
        C=d["hist_C"][-n_in:].astype(np.float64),
        psd_plant=d["hist_psd"][-n_in:].astype(np.float64),
        L_plant_um=d["L_mid"].astype(np.float64) * 1e6,
        t_end=float(t[-1]),
        warmup_seconds=float(d.get("warmup_seconds", t[-1])),
        source="preheat_cache",
    )
    if t_end_expected is not None and abs(out["t_end"] - t_end_expected) > 1.0:
        raise ValueError(
            f"预热缓存末时刻 {out['t_end']:.1f}s 与快照 t_window_end_s="
            f"{t_end_expected:.1f}s 不一致；请用快照 buf_* 或匹配的 preheat 缓存。")
    return out


def interp_psd_to_don(psd_plant: np.ndarray, L_plant_um: np.ndarray,
                      L_don_um: np.ndarray) -> np.ndarray:
    return np.stack([
        np.interp(L_don_um, L_plant_um, row, left=0.0, right=0.0)
        for row in psd_plant
    ]).astype(np.float64)


def number_mean_um(L_um: np.ndarray, psd: np.ndarray) -> float:
    mu0 = np.trapezoid(psd, L_um)
    if mu0 < 1e-30:
        return 0.0
    return float(np.trapezoid(psd * L_um, L_um) / mu0)


def _branches_replay(predictor: Predictor, T_seq, C_seq, psd_seq,
                    future_T: np.ndarray, n_windows: int) -> np.ndarray:
    """按推理滚动过程重建各窗 Branch。"""
    n_out = predictor.n_out
    T_seq = T_seq.copy()
    C_seq = C_seq.copy()
    psd_seq = psd_seq.copy()
    branches = []
    for w in range(n_windows):
        i0 = w * n_out
        T_f = np.asarray(future_T[i0:i0 + n_out], dtype=np.float64)
        branches.append(predictor._build_branch_from_seq(
            T_seq, C_seq, psd_seq, T_f))
        pred_psd, pred_conc = predictor._forward_branch(branches[-1], True)
        T_seq = np.concatenate([T_seq[n_out:], T_f])
        C_seq = np.concatenate([C_seq[n_out:], pred_conc])
        psd_seq = np.concatenate([psd_seq[n_out:], pred_psd], axis=0)
    return np.stack(branches, axis=0)


def _plot_result(inp: dict, res: dict, future_T: np.ndarray,
                 out_dir: str, title: str, dt: float):
    os.makedirs(out_dir, exist_ok=True)
    L = res["L_eval"]
    t_in = res["ctx_t"] / 60.0
    t_pred = res["t_abs"] / 60.0
    split = res["t_input_end"] / 60.0

    # 温度：历史 + 计划
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(inp["t"] / 60.0, inp["T"], "b-", lw=1.5, label="输入窗历史 T")
    t_plan = res["t_input_end"] + np.arange(1, len(future_T) + 1) * dt
    ax.plot(t_plan / 60.0, future_T, "r--", lw=1.5, label="计划 future_T")
    ax.axvline(split, color="gray", ls=":", lw=1.2)
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("T (K)")
    ax.set_title("Branch 输入：预热末 15min + 计划温度")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "temperature.png"), dpi=150)
    plt.close(fig)

    # 浓度
    fig, ax = plt.subplots(figsize=(9, 4))
    full_t = np.concatenate([t_in, t_pred])
    ax.plot(full_t, np.concatenate([res["ctx_conc"], res["pred_conc"]]),
            "g-", lw=1.2)
    ax.axvline(split, color="k", ls=":", lw=1)
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("C")
    ax.set_title(title + " — 浓度")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "concentration.png"), dpi=150)
    plt.close(fig)

    # PSD 时空 + 末时刻
    full_psd = np.concatenate([res["ctx_psd"], res["pred_psd"]], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    extent = [full_t[0], full_t[-1], L[0], L[-1]]
    im = axes[0].imshow(full_psd.T, aspect="auto", origin="lower",
                        extent=extent, cmap="viridis")
    axes[0].axvline(split, color="w", ls="--", lw=1)
    axes[0].set_xlabel("时间 (min)")
    axes[0].set_ylabel("L (μm)")
    axes[0].set_title("PSD（输入段+预测段）")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    Lm = [number_mean_um(L, row) for row in res["pred_psd"]]
    axes[1].plot(t_pred, Lm, "b-o", ms=3, lw=1.2)
    axes[1].set_xlabel("预测时间 (min)")
    axes[1].set_ylabel("L_mean (μm)")
    axes[1].set_title("数均粒径（预测段）")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "psd.png"), dpi=150)
    plt.close(fig)


def main():
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _nmpc = os.path.join(os.path.dirname(_root), "NMPC_Modular")
    default_snap = os.path.join(_nmpc, "cache", "first_opt_snapshot.npz")
    default_preheat = None

    cfg = get_default_config()
    ap = argparse.ArgumentParser(
        description="预热末15min + 计划温度 → DON 推理")
    ap.add_argument("--opt_snapshot", default=default_snap,
                    help="含 buf_* 与 future_T 的首次优化快照")
    ap.add_argument("--preheat_cache", default=default_preheat,
                    help="可选：从预热缓存取输入窗（须与快照 t_window_end_s 一致）")
    ap.add_argument("--run", default=None,
                    help="训练 run 目录或 weights/best.pt 路径，默认最新")
    ap.add_argument("--npz", default=cfg.path.npz_path)
    ap.add_argument("--n_windows", type=int, default=None,
                    help="自回归窗数，默认与快照一致或 3")
    ap.add_argument("--pred_min", type=float, default=None,
                    help="预测时长 [min]（覆盖 n_windows）")
    args = ap.parse_args()

    if not os.path.isfile(args.opt_snapshot):
        raise SystemExit(
            f"未找到优化快照: {args.opt_snapshot}\n"
            f"请先运行 NMPC：预热 → 第一次优化。")

    snap = np.load(args.opt_snapshot, allow_pickle=False)
    future_T = snap["future_T"].astype(np.float64)

    device = setup_device()
    run_dir = resolve_run_dir(args.run, cfg.path.results_dir)
    predictor = Predictor(run_dir, args.npz, cfg, device)
    n_in, n_out, dt = predictor.win.n_in, predictor.n_out, predictor.dt

    if args.pred_min is not None:
        n_windows = int(round(args.pred_min * 60.0 / (n_out * dt)))
    elif args.n_windows is not None:
        n_windows = args.n_windows
    elif "n_windows" in snap.files:
        n_windows = int(snap["n_windows"])
    else:
        n_windows = 3

    need_T = n_windows * n_out
    if future_T.size < need_T:
        raise SystemExit(
            f"快照 future_T 仅 {future_T.size} 点，需要 {need_T} 点。"
            f"请增大 --n_windows 或重新保存快照。")
    future_T = future_T[:need_T].copy()

    t_end_snap = float(snap["t_window_end_s"]) if "t_window_end_s" in snap.files else (
        float(snap["buf_t"][-1]) if "buf_t" in snap.files else None)
    if args.preheat_cache:
        if not os.path.isfile(args.preheat_cache):
            raise SystemExit(f"未找到预热缓存: {args.preheat_cache}")
        inp = load_preheat_input(
            args.preheat_cache, n_in, dt, t_end_expected=t_end_snap)
    else:
        inp = load_input_from_snapshot(args.opt_snapshot)
    psd_don = interp_psd_to_don(
        inp["psd_plant"], inp["L_plant_um"], predictor.L_full)

    print(f"[10] DON run: {run_dir}")
    print(f"     输入源: {inp.get('source', 'snapshot')}"
          + (f" ({args.preheat_cache})" if args.preheat_cache else ""))
    print(f"     快照: {args.opt_snapshot}")
    print(f"     输入窗 t={inp['t'][0]/60:.1f}~{inp['t'][-1]/60:.1f} min  "
          f"T_end={inp['T'][-1]:.2f}K")
    print(f"     计划 T: {need_T} 点  预测 {n_windows}×5min")

    branches = _branches_replay(
        predictor, inp["T"], inp["C"], psd_don.copy(), future_T, n_windows)
    print(f"     Branch 形状: {branches.shape}  (n_win, branch_dim)")

    if "branches" in snap.files:
        ref = snap["branches"][:n_windows].astype(np.float32)
        br_err = np.max(np.abs(branches.astype(np.float32) - ref))
        print(f"     vs NMPC 快照 Branch max|diff| = {br_err:.3e}")

    res = predictor.rolling_predict_autoregressive_buffer(
        inp["T"], inp["C"], psd_don.copy(),
        n_windows=n_windows,
        future_T=future_T,
        t_input_end=inp["t_end"],
    )

    if "pred_psd" in snap.files and int(snap.get("n_windows", n_windows)) >= n_windows:
        ref_psd = snap["pred_psd"].astype(np.float64)
        n_ref = min(ref_psd.shape[0], res["pred_psd"].shape[0])
        psd_err = np.max(np.abs(res["pred_psd"][:n_ref] - ref_psd[:n_ref]))
        psd_den = max(float(np.max(np.abs(ref_psd[:n_ref]))), 1e-30)
        print(f"     vs NMPC 快照 PSD max|diff|/max = {psd_err/psd_den:.3e}")

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(run_dir, "infer_preheat_planned", tag)
    os.makedirs(out_dir, exist_ok=True)

    _plot_result(inp, res, future_T, out_dir,
                 f"DON 推理 {n_windows}×5min", dt)

    np.savez_compressed(
        os.path.join(out_dir, "inference.npz"),
        branches=branches.astype(np.float32),
        future_T=future_T,
        input_t=inp["t"],
        input_T=inp["T"],
        input_C=inp["C"],
        input_psd_don=psd_don,
        L_don=predictor.L_full,
        pred_t=res["t_abs"],
        pred_psd=res["pred_psd"],
        pred_conc=res["pred_conc"],
        T_plan=res["T_plan"],
        t_input_end=inp["t_end"],
        n_windows=n_windows,
    )

    meta = dict(
        run_dir=run_dir,
        preheat_cache=args.preheat_cache,
        opt_snapshot=args.opt_snapshot,
        n_windows=n_windows,
        branch_dim=int(branches.shape[1]),
        L_mean_end=number_mean_um(res["L_eval"], res["pred_psd"][-1]),
        T_plan_end=float(res["T_plan"][-1]),
    )
    if "u_seq" in snap.files:
        meta["u_seq_K_per_h"] = (snap["u_seq"].astype(float) * 3600).tolist()
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    print(f"[OK] 推理结果: {out_dir}")
    print(f"     L_mean_end = {meta['L_mean_end']:.2f} μm")
    print("     - inference.npz  temperature.png  concentration.png  psd.png")


if __name__ == "__main__":
    main()
