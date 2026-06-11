"""
脚本 09：可视化 35min 预热缓存中「后 15min」输入窗。

展示 DON 初始输入段（预热末 15min）的：
  1. 全粒度 PSD 三维演化曲面（时间 × L × n(L)）
  2. 温度 / 浓度随时间曲线
  3. PSD 时空热力图（附）

用法::
    python scripts/09_visualize_preheat_input.py
    python scripts/09_visualize_preheat_input.py --preheat_cache ../NMPC_Modular/cache/preheat_2100s.npz
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                          "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_last_15min(cache_path: str, input_seconds: float = 900.0,
                    dt: float = 3.0) -> dict:
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"未找到预热缓存: {cache_path}")
    d = np.load(cache_path, allow_pickle=False)
    n_in = int(round(input_seconds / dt))
    if d["hist_t"].size < n_in:
        raise ValueError(
            f"缓存仅 {d['hist_t'].size} 点，不足 15min ({n_in} 点)。")

    t = d["hist_t"][-n_in:].astype(np.float64)
    T = d["hist_T"][-n_in:].astype(np.float64)
    C = d["hist_C"][-n_in:].astype(np.float64)
    psd = d["hist_psd"][-n_in:].astype(np.float64)
    L_um = d["L_mid"].astype(np.float64) * 1e6
    warmup = float(d.get("warmup_seconds", d["hist_t"][-1]))

    if not np.allclose(np.diff(t), dt, atol=1e-2):
        raise ValueError(f"时间轴须均匀间隔 {dt}s。")

    return dict(
        t=t, T=T, C=C, psd=psd, L_um=L_um,
        t_min=t / 60.0,
        warmup_min=warmup / 60.0,
        t_start_min=t[0] / 60.0,
        t_end_min=t[-1] / 60.0,
        rc_kph=float(d["hist_rc"][-1]) * 3600.0 if "hist_rc" in d else None,
    )


def number_mean_um(L_um: np.ndarray, psd: np.ndarray) -> np.ndarray:
    out = np.zeros(psd.shape[0], dtype=np.float64)
    for i, row in enumerate(psd):
        mu0 = np.trapezoid(row, L_um)
        if mu0 > 1e-30:
            out[i] = np.trapezoid(row * L_um, L_um) / mu0
    return out


def save_figures(data: dict, out_dir: str, l_step: int = 8):
    os.makedirs(out_dir, exist_ok=True)
    t_min = data["t_min"]
    L_um = data["L_um"]
    psd = data["psd"]
    T, C = data["T"], data["C"]
    L_plot = L_um[::l_step]
    psd_plot = psd[:, ::l_step]
    Tg, Lg = np.meshgrid(t_min, L_plot)
    Z = psd_plot.T

    meta = (f"预热 {data['warmup_min']:.0f}min 末段  "
            f"t = {data['t_start_min']:.1f}~{data['t_end_min']:.1f} min")
    if data.get("rc_kph") is not None:
        meta += f"  |  rc ≈ {data['rc_kph']:.3f} K/h"

    # ---- 1. 主面板：3D PSD + T/C ----
    fig = plt.figure(figsize=(14, 5.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(
        Tg, Lg, Z, cmap="viridis", linewidth=0,
        antialiased=True, alpha=0.92)
    ax3d.set_xlabel("时间 (min)")
    ax3d.set_ylabel("L (μm)")
    ax3d.set_zlabel("n(L)")
    ax3d.set_title(f"PSD 三维演化（DON 输入窗 15min）\n{meta}")
    fig.colorbar(surf, ax=ax3d, shrink=0.55, pad=0.08, label="n(L)")

    ax_tc = fig.add_subplot(1, 2, 2)
    ax_tc.plot(t_min, T, "r-", lw=1.8, label="温度 T (K)")
    ax_tc.set_xlabel("时间 (min)")
    ax_tc.set_ylabel("温度 T (K)", color="r")
    ax_tc.tick_params(axis="y", labelcolor="r")
    ax_tc.grid(alpha=0.3)

    ax_c = ax_tc.twinx()
    ax_c.plot(t_min, C, "b-", lw=1.8, label="浓度 C")
    ax_c.set_ylabel("浓度 C", color="b")
    ax_c.tick_params(axis="y", labelcolor="b")

    Lm = number_mean_um(L_um, psd)
    ax_lm = ax_tc.twinx()
    ax_lm.spines["right"].set_position(("axes", 1.12))
    ax_lm.plot(t_min, Lm, "g--", lw=1.4, label="L_mean (μm)")
    ax_lm.set_ylabel("L_mean (μm)", color="g")
    ax_lm.tick_params(axis="y", labelcolor="g")

    ax_tc.set_title("温度 / 浓度 / 数均粒径")
    h1, l1 = ax_tc.get_legend_handles_labels()
    h2, l2 = ax_c.get_legend_handles_labels()
    h3, l3 = ax_lm.get_legend_handles_labels()
    ax_tc.legend(h1 + h2 + h3, l1 + l2 + l3, loc="upper right", fontsize=8)

    fig.suptitle("35min 预热 → 后 15min 输入数据", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "preheat_input_15min_panel.png"), dpi=160)
    plt.close(fig)

    # ---- 2. 时空热力图 ----
    fig, ax = plt.subplots(figsize=(10, 4.5))
    extent = [t_min[0], t_min[-1], L_um[0], L_um[-1]]
    im = ax.imshow(psd.T, aspect="auto", origin="lower",
                   extent=extent, cmap="viridis")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("L (μm)")
    ax.set_title(f"PSD 时空热力图（后 15min）\n{meta}")
    fig.colorbar(im, ax=ax, fraction=0.046, label="n(L)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "preheat_input_spacetime.png"), dpi=160)
    plt.close(fig)

    # ---- 3. 多时刻 PSD 剖面 ----
    n_snap = 5
    idx = np.linspace(0, len(t_min) - 1, n_snap, dtype=int)
    fig, axes = plt.subplots(1, n_snap, figsize=(14, 3.8), sharey=True)
    for ax, k in zip(axes, idx):
        ax.plot(L_um, psd[k], "b-", lw=1.4)
        ax.set_title(f"t={t_min[k]:.1f} min")
        ax.set_xlabel("L (μm)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("n(L)")
    fig.suptitle("后 15min 内多时刻 PSD 分布", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "preheat_input_psd_snapshots.png"), dpi=160)
    plt.close(fig)

    # ---- 4. 单独 T / C 大图 ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t_min, T, "r-", lw=1.8)
    axes[0].set_ylabel("T (K)")
    axes[0].set_title("温度")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t_min, C, "b-", lw=1.8)
    axes[1].set_xlabel("时间 (min)")
    axes[1].set_ylabel("C")
    axes[1].set_title("浓度")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"后 15min 温度与浓度  ({meta})", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "preheat_input_T_C.png"), dpi=160)
    plt.close(fig)


def main():
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _nmpc = os.path.join(os.path.dirname(_root), "NMPC_Modular")
    default_cache = os.path.join(_nmpc, "cache", "preheat_2100s.npz")

    ap = argparse.ArgumentParser(description="可视化预热末 15min 输入窗")
    ap.add_argument("--preheat_cache", default=default_cache)
    ap.add_argument("--input_min", type=float, default=15.0)
    ap.add_argument("--dt", type=float, default=3.0)
    ap.add_argument("--l_step", type=int, default=8,
                    help="3D 曲面粒径下采样步长")
    ap.add_argument("--out", default=None, help="输出目录")
    args = ap.parse_args()

    data = load_last_15min(
        args.preheat_cache, input_seconds=args.input_min * 60.0, dt=args.dt)

    if args.out:
        out_dir = args.out
    else:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(_root, "results", "preheat_input_viz", tag)

    save_figures(data, out_dir, l_step=args.l_step)

    print(f"[09] 预热缓存: {args.preheat_cache}")
    print(f"     输入窗: t={data['t_start_min']:.1f}~{data['t_end_min']:.1f} min "
          f"({data['psd'].shape[0]} 点)")
    print(f"     T: {data['T'][0]:.2f} → {data['T'][-1]:.2f} K")
    print(f"     C: {data['C'][0]:.5f} → {data['C'][-1]:.5f}")
    print(f"     L_mean: {number_mean_um(data['L_um'], data['psd'])[0]:.1f} → "
          f"{number_mean_um(data['L_um'], data['psd'])[-1]:.1f} μm")
    print(f"[OK] 图已保存: {out_dir}")
    print("     - preheat_input_15min_panel.png   （3D PSD + T/C/L_mean）")
    print("     - preheat_input_spacetime.png")
    print("     - preheat_input_psd_snapshots.png")
    print("     - preheat_input_T_C.png")


if __name__ == "__main__":
    main()
