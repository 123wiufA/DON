"""
脚本 07：自回归滚动预测。

与 05_rolling_predict.py（teacher forcing）的区别：
  - 05：每个窗口都用仿真「真实前 15min」作输入
  - 07：第 1 窗用真实 15min；之后每窗把「预测的 5min 浓度/PSD」滚入输入窗

每窗 Branch 含输出段计划温度 T_future（50 点）：
  - ``true``：本窗 T_future 用仿真真值（C/PSD 自回归）
  - ``hold``：T_future 保持输入末点（恒温计划）

用法::
    python scripts/07_autoregressive_rolling.py --case CR_1_08
    python scripts/07_autoregressive_rolling.py --case CR_1_08 --n_windows 6
    python scripts/07_autoregressive_rolling.py --case CR_1_08 --T_source hold
    python scripts/07_autoregressive_rolling.py --case CR_1_08 --compare_tf
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


def _save_plots(res: dict, out_dir: str, tag: str):
    """保存与 05 类似的全粒度图。"""
    t_min = res["t_abs"] / 60.0
    L = res["L_eval"]
    true_psd, pred_psd = res["true_psd"], res["pred_psd"]
    in_dur_min = res["split_seconds"] / 60.0
    ctx_tmin = res["ctx_t"] / 60.0
    ctx_psd, ctx_conc = res["ctx_psd"], res["ctx_conc"]
    split_abs = ctx_tmin[0] + in_dur_min if ctx_tmin.size else in_dur_min

    full_tmin = np.concatenate([ctx_tmin, t_min])
    full_true_psd = np.concatenate([ctx_psd, true_psd], axis=0)
    full_pred_psd = np.concatenate([ctx_psd, pred_psd], axis=0)
    full_true_conc = np.concatenate([ctx_conc, res["true_conc"]])

    plt.figure(figsize=(10, 5))
    plt.plot(full_tmin, full_true_conc, "b-", lw=1.4, label="仿真")
    plt.plot(t_min, res["pred_conc"], "r--", lw=1.4, label="自回归预测")
    plt.axvspan(ctx_tmin[0], split_abs, color="0.85", alpha=0.6, label="初始输入段")
    plt.axvline(split_abs, color="k", ls=":", lw=1.2)
    plt.xlabel("时间 (min)"); plt.ylabel("浓度 C")
    plt.title(f"浓度：自回归滚动 vs 仿真 [{res['case']}] {tag}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "conc_full.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    extent = [full_tmin[0], full_tmin[-1], L[0], L[-1]]
    err = full_pred_psd - full_true_psd
    for ax, title, grid in zip(axes,
                               ["仿真", "自回归预测", "误差"],
                               [full_true_psd.T, full_pred_psd.T, err.T]):
        im = ax.imshow(grid, aspect="auto", origin="lower", extent=extent,
                       cmap="viridis")
        ax.axvline(split_abs, color="w", ls="--", lw=1.2)
        ax.set_xlabel("时间 (min)"); ax.set_ylabel("L (μm)"); ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"PSD 时空（自回归，白虚线=输入/预测分界）[{res['case']}] {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spacetime_full.png"), dpi=150)
    plt.close(fig)

    n_snap = 6
    snap_idx = np.linspace(0, len(full_tmin) - 1, n_snap, dtype=int)
    n_ctx = len(ctx_tmin)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, k in zip(axes.ravel(), snap_idx):
        is_input = k < n_ctx
        ax.plot(L, full_true_psd[k], "b-", lw=1.6, label="仿真")
        if not is_input:
            ax.plot(L, full_pred_psd[k], "r--", lw=1.6, label="自回归预测")
        seg = "输入段" if is_input else "预测段"
        ax.set_title(f"t = {full_tmin[k]:.1f} min  ({seg})")
        ax.set_xlabel("L (μm)"); ax.set_ylabel("n(L)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(f"PSD 分布：自回归滚动 vs 仿真 [{res['case']}]", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "psd_snapshots_full.png"), dpi=150)
    plt.close(fig)

    peak_idx = int(np.argmax(true_psd.mean(axis=0)))
    plt.figure(figsize=(10, 5))
    plt.plot(full_tmin, full_true_psd[:, peak_idx], "b-", lw=1.4, label="仿真")
    plt.plot(t_min, pred_psd[:, peak_idx], "r--", lw=1.4, label="自回归预测")
    plt.axvspan(ctx_tmin[0], split_abs, color="0.85", alpha=0.6, label="初始输入段")
    plt.axvline(split_abs, color="k", ls=":", lw=1.2)
    plt.xlabel("时间 (min)")
    plt.ylabel(f"n(L={L[peak_idx]:.1f}μm, t)")
    plt.title(f"峰值粒径 PSD 随时间（自回归）[{res['case']}]")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psd_peak_track.png"), dpi=150)
    plt.close()


def main():
    cfg = get_default_config()
    ap = argparse.ArgumentParser(description="自回归滚动预测")
    ap.add_argument("--case", required=True, help="工况名，如 CR_1_08")
    ap.add_argument("--start_sec", type=float, default=0.0,
                    help="初始输入窗起点 [s]，默认 0")
    ap.add_argument("--n_windows", type=int, default=None,
                    help="滚动窗数，默认直到数据结束")
    ap.add_argument("--T_source", choices=("true", "hold"), default="true",
                    help="更新输入窗时温度来源：true=仿真真值, hold=保持末点")
    ap.add_argument("--compare_tf", action="store_true",
                    help="同时跑 teacher forcing (05) 并打印误差对比")
    ap.add_argument("--run", default=None, help="结果目录，默认取最新")
    ap.add_argument("--npz", default=cfg.path.npz_path)
    ap.add_argument("--eval_grid", action="store_true",
                    help="仅用评估下采样网格（默认全粒度）")
    args = ap.parse_args()

    device = setup_device()
    run_dir = args.run or latest_run(cfg.path.results_dir)
    print(f"[AR-Rolling] 结果目录: {run_dir}")

    predictor = Predictor(run_dir, args.npz, cfg, device)
    full_res = not args.eval_grid

    res = predictor.rolling_predict_autoregressive(
        args.case,
        start_seconds=args.start_sec,
        n_windows=args.n_windows,
        T_source=args.T_source,
        full_resolution=full_res,
    )

    t_min = res["t_abs"] / 60.0
    rl2_psd = relative_l2(res["pred_psd"], res["true_psd"])
    rl2_c = relative_l2(res["pred_conc"], res["true_conc"])
    grid_tag = "全粒度" if res["full_resolution"] else "评估网格"
    print(f"[AR-Rolling] 工况 {args.case}: {res['n_windows']} 窗, "
          f"T_source={res['T_source']}, 粒径={res['L_eval'].size}({grid_tag})")
    print(f"  覆盖 {t_min[0]:.1f}~{t_min[-1]:.1f} min")
    print(f"  PSD Rel.L2 = {rl2_psd:.4e}   浓度 Rel.L2 = {rl2_c:.4e}")

    if args.compare_tf:
        tf = predictor.rolling_predict(
            args.case, full_resolution=full_res)
        # 对齐相同预测时段
        n_pts = min(len(res["pred_psd"]), len(tf["pred_psd"]))
        rl2_tf_psd = relative_l2(tf["pred_psd"][:n_pts], res["true_psd"][:n_pts])
        rl2_tf_c = relative_l2(tf["pred_conc"][:n_pts], res["true_conc"][:n_pts])
        print(f"[Compare] Teacher forcing (05) 同段 PSD Rel.L2 = {rl2_tf_psd:.4e}  "
              f"Conc = {rl2_tf_c:.4e}")
        print(f"          自回归 / TF 误差比 PSD = {rl2_psd / max(rl2_tf_psd, 1e-30):.2f}x")

    tag = f"T={args.T_source}"
    out_dir = os.path.join(run_dir, "rolling_ar", args.case, tag)
    os.makedirs(out_dir, exist_ok=True)
    _save_plots(res, out_dir, tag)

    print(f"[OK] 自回归滚动预测图已保存到 {out_dir}")
    print("     - spacetime_full.png / conc_full.png / psd_snapshots_full.png / psd_peak_track.png")


if __name__ == "__main__":
    main()
