"""
脚本 05：全程滚动预测。

在整条工况上滚动：每次以「真实的前 15min」为输入，预测随后 5min，
默认步长 = 5min（输出段首尾相接），从而拼接出从 15min 到结尾的完整预测轨迹，
并与仿真真值逐点对比、出图、统计误差。

用法::
    python scripts/05_rolling_predict.py --case CR_1_13
    python scripts/05_rolling_predict.py --case CR_1_08 --stride_min 5 --run results/2026xxxx_xxxxxx
"""

import os
import sys
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from donpbe.config import get_default_config
from donpbe.device import setup_device
from donpbe.predictor import Predictor
from donpbe.utils import relative_l2


def latest_run(results_dir: str) -> str:
    runs = [d for d in glob.glob(os.path.join(results_dir, "*"))
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "weights", "best.pt"))]
    if not runs:
        raise FileNotFoundError(f"{results_dir} 下没有可用的训练结果。")
    return max(runs, key=os.path.getmtime)


def main():
    cfg = get_default_config()
    ap = argparse.ArgumentParser(description="全程滚动预测")
    ap.add_argument("--case", required=True, help="工况名，如 CR_1_13")
    ap.add_argument("--stride_min", type=float, default=None,
                    help="滚动步长（分钟），默认=输出窗口 5min（无缝覆盖）")
    ap.add_argument("--run", default=None, help="结果目录，默认取最新")
    ap.add_argument("--npz", default=cfg.path.npz_path)
    ap.add_argument("--eval_grid", action="store_true",
                    help="仅用评估下采样网格（默认用完整粒径网格做全粒度预测）")
    args = ap.parse_args()

    device = setup_device()
    run_dir = args.run or latest_run(cfg.path.results_dir)
    print(f"[Rolling] 结果目录: {run_dir}")

    predictor = Predictor(run_dir, args.npz, cfg, device)
    stride_sec = args.stride_min * 60.0 if args.stride_min else None
    res = predictor.rolling_predict(args.case, stride_seconds=stride_sec,
                                    full_resolution=not args.eval_grid)

    t_min = res["t_abs"] / 60.0
    L = res["L_eval"]
    true_psd, pred_psd = res["true_psd"], res["pred_psd"]
    rl2_psd = relative_l2(pred_psd, true_psd)
    rl2_c = relative_l2(res["pred_conc"], res["true_conc"])
    grid_tag = "全粒度" if res["full_resolution"] else "评估网格"
    print(f"[Rolling] 工况 {args.case}: 共 {res['n_windows']} 个窗口, "
          f"覆盖 {t_min[0]:.1f}~{t_min[-1]:.1f}min, "
          f"粒径点数={L.size}({grid_tag})")
    print(f"  全程 PSD Rel.L2 = {rl2_psd:.4e}   浓度 Rel.L2 = {rl2_c:.4e}")

    out_dir = os.path.join(run_dir, "rolling", args.case)
    os.makedirs(out_dir, exist_ok=True)

    # 输入段(0~15min)上下文
    split_min = res["split_seconds"] / 60.0
    ctx_tmin = res["ctx_t"] / 60.0
    ctx_psd, ctx_conc = res["ctx_psd"], res["ctx_conc"]
    # 拼接「输入段 + 预测段」的完整时间轴（横轴从 0min 起）
    full_tmin = np.concatenate([ctx_tmin, t_min])
    full_true_psd = np.concatenate([ctx_psd, true_psd], axis=0)
    # 预测图在输入段用真值填充（输入段是已知条件，非预测）
    full_pred_psd = np.concatenate([ctx_psd, pred_psd], axis=0)
    full_true_conc = np.concatenate([ctx_conc, res["true_conc"]])
    full_pred_conc = np.concatenate([ctx_conc, res["pred_conc"]])

    # ---- 全程浓度曲线（含输入段 + 分界线）----
    plt.figure(figsize=(10, 5))
    plt.plot(full_tmin, full_true_conc, "b-", lw=1.4, label="仿真")
    plt.plot(t_min, res["pred_conc"], "r--", lw=1.4, label="预测")
    plt.axvspan(0, split_min, color="0.85", alpha=0.6, label="输入段(0~15min)")
    plt.axvline(split_min, color="k", ls=":", lw=1.2)
    plt.xlabel("时间 (min)"); plt.ylabel("浓度 C")
    plt.title(f"全程浓度：滚动预测 vs 仿真 [{args.case}]")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "conc_full.png"), dpi=150)
    plt.close()

    # ---- 全粒度全程 PSD 时空热力图（含输入段 + 分界线）----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    err = full_pred_psd - full_true_psd
    grids = [full_true_psd.T, full_pred_psd.T, err.T]
    titles = ["仿真", "预测(输入段=真值)", "误差(预测-仿真)"]
    extent = [full_tmin[0], full_tmin[-1], L[0], L[-1]]
    for ax, title, grid in zip(axes, titles, grids):
        im = ax.imshow(grid, aspect="auto", origin="lower", extent=extent,
                       cmap="viridis")
        ax.axvline(split_min, color="w", ls="--", lw=1.2)
        ax.set_xlabel("时间 (min)"); ax.set_ylabel("L (μm)"); ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"PSD 时空演化（白虚线=输入/预测分界 {split_min:.0f}min）[{args.case}]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spacetime_full.png"), dpi=150)
    plt.close(fig)

    # ---- 全粒度多时刻 PSD 分布快照（含输入段时刻）----
    n_snap = 6
    snap_idx = np.linspace(0, len(full_tmin) - 1, n_snap, dtype=int)
    n_ctx = len(ctx_tmin)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, k in zip(axes.ravel(), snap_idx):
        is_input = k < n_ctx
        ax.plot(L, full_true_psd[k], "b-", lw=1.6, label="仿真")
        if not is_input:                       # 预测段才画预测线
            ax.plot(L, full_pred_psd[k], "r--", lw=1.6, label="预测")
        seg = "输入段" if is_input else "预测段"
        ax.set_title(f"t = {full_tmin[k]:.1f} min  ({seg})")
        ax.set_xlabel("L (μm)"); ax.set_ylabel("n(L)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(f"全粒度 PSD 分布：滚动预测 vs 仿真 [{args.case}]", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "psd_snapshots_full.png"), dpi=150)
    plt.close(fig)

    # ---- 关键粒径处的 PSD 时间曲线（含输入段 + 分界线）----
    peak_idx = int(np.argmax(true_psd.mean(axis=0)))
    plt.figure(figsize=(10, 5))
    plt.plot(full_tmin, full_true_psd[:, peak_idx], "b-", lw=1.4, label="仿真")
    plt.plot(t_min, pred_psd[:, peak_idx], "r--", lw=1.4, label="预测")
    plt.axvspan(0, split_min, color="0.85", alpha=0.6, label="输入段(0~15min)")
    plt.axvline(split_min, color="k", ls=":", lw=1.2)
    plt.xlabel("时间 (min)")
    plt.ylabel(f"n(L={L[peak_idx]:.1f}μm, t)")
    plt.title(f"全程滚动预测：峰值粒径处 PSD 随时间 [{args.case}]")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psd_peak_track.png"), dpi=150)
    plt.close()

    print(f"[OK] 全粒度全程滚动预测图已保存到 {out_dir}")
    print("     - spacetime_full.png    （全粒度全程时空热力图：真值/预测/误差）")
    print("     - psd_snapshots_full.png（全粒度多时刻 PSD 分布对比）")
    print("     - psd_peak_track.png    （峰值粒径处 PSD 随时间）")
    print("     - conc_full.png         （全程浓度曲线）")


if __name__ == "__main__":
    main()
