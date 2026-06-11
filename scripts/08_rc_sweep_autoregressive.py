"""
脚本 08：35min 预热末 15min 作输入，30 种降温速率下 90min 自回归预测。

流程：
  1. 读取 NMPC 预热缓存（默认 35min Plant 仿真）
  2. 取最后 15min 的 T/C/PSD，PSD 插值到 DON 粒径轴
  3. rc = 1~15 K/h 共 30 点，各做 18 窗 × 5min = 90min 自回归预测
  4. 保存汇总图与逐工况图

用法::
    python scripts/08_rc_sweep_autoregressive.py
    python scripts/08_rc_sweep_autoregressive.py --preheat_cache ../NMPC_Modular/cache/preheat_2100s.npz
    python scripts/08_rc_sweep_autoregressive.py --run results/20260611_095626
"""

import os
import sys
import glob
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


def latest_run(results_dir: str) -> str:
    runs = [d for d in glob.glob(os.path.join(results_dir, "*"))
            if os.path.isdir(d) and os.path.isfile(
                os.path.join(d, "weights", "best.pt"))]
    if not runs:
        raise FileNotFoundError(f"{results_dir} 下没有可用的训练结果。")
    return max(runs, key=os.path.getmtime)


def load_input_from_preheat(cache_path: str, n_in: int, dt: float) -> dict:
    """从预热缓存取最后 15min 输入窗。"""
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"未找到预热缓存: {cache_path}")
    d = np.load(cache_path, allow_pickle=False)
    warmup = float(d.get("warmup_seconds", d["hist_t"][-1]))
    if d["hist_t"].size < n_in:
        raise ValueError(
            f"缓存采样点 {d['hist_t'].size} < 输入窗 {n_in} 点。")

    t = d["hist_t"][-n_in:].astype(np.float64)
    if not np.allclose(np.diff(t), dt, atol=1e-2):
        raise ValueError(f"缓存时间轴须均匀 {dt}s。")

    return dict(
        t=t,
        T=d["hist_T"][-n_in:].astype(np.float64),
        C=d["hist_C"][-n_in:].astype(np.float64),
        psd_plant=d["hist_psd"][-n_in:].astype(np.float64),
        L_plant_um=d["L_mid"].astype(np.float64) * 1e6,
        t_end=float(t[-1]),
        warmup_seconds=warmup,
    )


def interp_psd_to_don(psd_plant: np.ndarray, L_plant_um: np.ndarray,
                      L_don_um: np.ndarray) -> np.ndarray:
    rows = [np.interp(L_don_um, L_plant_um, row, left=0.0, right=0.0)
            for row in psd_plant]
    return np.stack(rows, axis=0).astype(np.float64)


def plan_T_const_rc(T0: float, rc_k_per_s: float, dt: float,
                    n_pts: int) -> np.ndarray:
    """恒定降温速率，每 dt 一点（与 NMPC 温度规划一致）。"""
    T = np.empty(n_pts, dtype=np.float64)
    T_curr = float(T0)
    for i in range(n_pts):
        T_curr -= rc_k_per_s * dt
        T[i] = T_curr
    return T


def number_mean_um(L_um: np.ndarray, psd: np.ndarray) -> float:
    mu0 = np.trapezoid(psd, L_um)
    if mu0 < 1e-30:
        return 0.0
    return float(np.trapezoid(psd * L_um, L_um) / mu0)


def _save_single_rc(res: dict, out_dir: str, rc_kph: float):
    os.makedirs(out_dir, exist_ok=True)
    L = res["L_eval"]
    t_in = res["ctx_t"] / 60.0
    t_pred = res["t_abs"] / 60.0
    split = res["t_input_end"] / 60.0

    full_t = np.concatenate([t_in, t_pred])
    full_psd = np.concatenate([res["ctx_psd"], res["pred_psd"]], axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(full_t, np.concatenate([res["ctx_conc"], res["pred_conc"]]),
            "g-", lw=1.2, label="预测浓度")
    ax.axvline(split, color="k", ls=":", lw=1)
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("C")
    ax.set_title(f"浓度  rc={rc_kph:.2f} K/h")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "conc.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    extent = [full_t[0], full_t[-1], L[0], L[-1]]
    im = axes[0].imshow(full_psd.T, aspect="auto", origin="lower",
                        extent=extent, cmap="viridis")
    axes[0].axvline(split, color="w", ls="--", lw=1)
    axes[0].set_xlabel("时间 (min)")
    axes[0].set_ylabel("L (μm)")
    axes[0].set_title(f"PSD 自回归  rc={rc_kph:.2f} K/h")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    peak = int(np.argmax(res["pred_psd"].mean(axis=0)))
    axes[1].plot(full_t, full_psd[:, peak], "b-", lw=1.2)
    axes[1].axvline(split, color="k", ls=":", lw=1)
    axes[1].set_xlabel("时间 (min)")
    axes[1].set_ylabel(f"n(L≈{L[peak]:.0f}μm)")
    axes[1].set_title("峰值粒径处 PSD")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "psd.png"), dpi=150)
    plt.close(fig)


def _save_overview(all_res: list, rc_list_kph: np.ndarray, out_dir: str):
    cmap = plt.colormaps["viridis"].resampled(len(rc_list_kph))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, (res, rc) in enumerate(zip(all_res, rc_list_kph)):
        c = cmap(i)
        t = res["t_abs"] / 60.0
        L = res["L_eval"]
        Lm = [number_mean_um(L, row) for row in res["pred_psd"]]
        axes[0].plot(t, Lm, color=c, lw=1.0)
        axes[1].plot(t, res["pred_conc"], color=c, lw=1.0)
        axes[2].plot(t, res["T_plan"], color=c, lw=1.0)

    axes[0].set_title("数均粒径 L_mean（预测段）")
    axes[0].set_xlabel("t (min)")
    axes[0].set_ylabel("L_mean (μm)")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("浓度 C（预测段）")
    axes[1].set_xlabel("t (min)")
    axes[1].set_ylabel("C")
    axes[1].grid(alpha=0.3)

    axes[2].set_title("规划温度 T（恒定 rc）")
    axes[2].set_xlabel("t (min)")
    axes[2].set_ylabel("T (K)")
    axes[2].grid(alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(rc_list_kph[0],
                                                    rc_list_kph[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("rc (K/h)")

    fig.suptitle("30 种降温速率 × 90min 自回归预测（输入=预热末 15min）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overview_traces.png"), dpi=160)
    plt.close(fig)

    # 末端 L_mean vs rc
    L_end = [number_mean_um(r["L_eval"], r["pred_psd"][-1]) for r in all_res]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rc_list_kph, L_end, "o-", lw=1.5, ms=4)
    ax.set_xlabel("降温速率 rc (K/h)")
    ax.set_ylabel("90min 末端 L_mean (μm)")
    ax.set_title("预测末端粒径 vs 降温速率")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Lmean_end_vs_rc.png"), dpi=150)
    plt.close(fig)

    # 末端 PSD 叠图（稀疏标注）
    fig, ax = plt.subplots(figsize=(9, 5))
    step = max(1, len(rc_list_kph) // 8)
    for i in range(0, len(all_res), step):
        r = all_res[i]
        rc = rc_list_kph[i]
        ax.plot(r["L_eval"], r["pred_psd"][-1], lw=1.0,
                label=f"{rc:.1f} K/h")
    ax.set_xlabel("L (μm)")
    ax.set_ylabel("n(L) @ 90min")
    ax.set_title("预测末端 PSD（部分 rc）")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "psd_end_snapshots.png"), dpi=150)
    plt.close(fig)


def main():
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _nmpc = os.path.join(os.path.dirname(_root), "NMPC_Modular")
    default_cache = os.path.join(_nmpc, "cache", "preheat_2100s.npz")

    cfg = get_default_config()
    ap = argparse.ArgumentParser(description="rc 扫描自回归预测")
    ap.add_argument("--preheat_cache", default=default_cache,
                    help="35min 预热缓存 npz")
    ap.add_argument("--run", default=None, help="DON 训练 run，默认最新")
    ap.add_argument("--npz", default=cfg.path.npz_path)
    ap.add_argument("--pred_min", type=float, default=90.0,
                    help="预测时长 [min]")
    ap.add_argument("--rc_min", type=float, default=1.0)
    ap.add_argument("--rc_max", type=float, default=15.0)
    ap.add_argument("--n_rc", type=int, default=30)
    ap.add_argument("--save_each", action="store_true",
                    help="为每个 rc 单独存子目录图")
    args = ap.parse_args()

    device = setup_device()
    run_dir = args.run or latest_run(cfg.path.results_dir)
    predictor = Predictor(run_dir, args.npz, cfg, device)
    n_in = predictor.win.n_in
    n_out = predictor.n_out
    dt = predictor.dt

    inp = load_input_from_preheat(args.preheat_cache, n_in, dt)
    psd_don = interp_psd_to_don(
        inp["psd_plant"], inp["L_plant_um"], predictor.L_full)

    n_windows = int(round(args.pred_min * 60.0 / (n_out * dt)))
    need_T = n_windows * n_out
    rc_list_kph = np.linspace(args.rc_min, args.rc_max, args.n_rc)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(run_dir, "rc_sweep_ar", tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[08] run={run_dir}")
    print(f"     预热缓存: {args.preheat_cache}")
    print(f"     输入窗: t={inp['t'][0]/60:.1f}~{inp['t'][-1]/60:.1f} min "
          f"(末点 T={inp['T'][-1]:.2f}K)")
    print(f"     rc: {args.rc_min}~{args.rc_max} K/h × {args.n_rc}  "
          f"预测 {args.pred_min}min = {n_windows} 窗")

    all_res = []
    for rc_kph in rc_list_kph:
        rc = rc_kph / 3600.0
        future_T = plan_T_const_rc(inp["T"][-1], rc, dt, need_T)
        res = predictor.rolling_predict_autoregressive_buffer(
            inp["T"], inp["C"], psd_don.copy(),
            n_windows=n_windows,
            future_T=future_T,
            t_input_end=inp["t_end"],
            rc_kph=float(rc_kph),
        )
        all_res.append(res)
        L_end = number_mean_um(res["L_eval"], res["pred_psd"][-1])
        print(f"  rc={rc_kph:5.2f} K/h  L_mean_end={L_end:7.1f} μm  "
              f"T_end={res['T_plan'][-1]:.2f}K")
        if args.save_each:
            sub = os.path.join(out_dir, f"rc_{rc_kph:.2f}_Kph")
            _save_single_rc(res, sub, rc_kph)

    _save_overview(all_res, rc_list_kph, out_dir)

    # 数值归档
    np.savez_compressed(
        os.path.join(out_dir, "rc_sweep_results.npz"),
        rc_kph=rc_list_kph,
        t_input_end=inp["t_end"],
        input_T=inp["T"],
        input_C=inp["C"],
        L_don=predictor.L_full,
        pred_t=np.stack([r["t_abs"] for r in all_res]),
        pred_conc=np.stack([r["pred_conc"] for r in all_res]),
        pred_psd=np.stack([r["pred_psd"] for r in all_res]),
        T_plan=np.stack([r["T_plan"] for r in all_res]),
    )

    print(f"[OK] 图与数据已保存: {out_dir}")
    print("     - overview_traces.png")
    print("     - Lmean_end_vs_rc.png")
    print("     - psd_end_snapshots.png")
    print("     - rc_sweep_results.npz")
    if args.save_each:
        print("     - rc_*/conc.png, psd.png")


if __name__ == "__main__":
    main()
