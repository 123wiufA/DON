"""
脚本 06：可视化滑动窗口切片数据。

按当前训练切片规则（前 15min 输入 + 后 5min 输出，步长 5min），
从指定工况中均匀选取 5 个窗口（含第一个与最后一个），展示：

  1. 窗口内全粒度 PSD 三维演化曲面（L × 时间 × n(L)）
  2. 窗口内温度 / 浓度随时间演化（标出输入段与输出段分界）

用法::
    python scripts/06_visualize_slices.py --case CR_1_01
    python scripts/06_visualize_slices.py --case CR_1_13 --out results/slice_inspect
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from donpbe.config import get_default_config

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def window_starts(n_time: int, n_window: int, stride_pts: int):
    """与 dataset.PBEWindowData._window_starts 相同的切片起点规则。"""
    last_start = n_time - n_window
    if last_start < 0:
        raise ValueError(f"时间点 {n_time} 小于窗口长度 {n_window}。")
    return list(range(0, last_start + 1, stride_pts))


def pick_slice_indices(n_slices: int, n_total: int):
    """均匀选取切片索引，保证包含首尾。"""
    if n_total <= n_slices:
        return list(range(n_total))
    return np.linspace(0, n_total - 1, n_slices, dtype=int).tolist()


def plot_slice_panel(case_name: str, slice_id: int, t0: int,
                     t_abs, L, psd_win, T_win, C_win,
                     n_in: int, save_path: str,
                     l_step: int = 8) -> None:
    """绘制单个切片：左 PSD 三维曲面，右 温度/浓度双轴曲线。"""
    t_min = t_abs / 60.0
    split_min = t_abs[n_in] / 60.0          # 输入/输出分界（绝对时刻）

    # 粒径下采样，避免 3D 曲面过密
    L_plot = L[::l_step]
    psd_plot = psd_win[:, ::l_step]                     # (n_window, n_L_sub)
    Tg, Lg = np.meshgrid(t_min, L_plot)                 # (n_L_sub, n_window)
    Z = psd_plot.T

    fig = plt.figure(figsize=(14, 5.2))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(
        Tg, Lg, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92)
    ax3d.set_xlabel("时间 (min)")
    ax3d.set_ylabel("L (μm)")
    ax3d.set_zlabel("n(L)")
    ax3d.set_title(f"PSD 三维演化  [{case_name}  切片#{slice_id}]")
    fig.colorbar(surf, ax=ax3d, shrink=0.55, pad=0.08, label="n(L)")

    ax_tc = fig.add_subplot(1, 2, 2)
    ax_tc.plot(t_min, T_win, "r-", lw=1.6, label="温度 T (K)")
    ax_tc.set_xlabel("时间 (min)")
    ax_tc.set_ylabel("温度 T (K)", color="r")
    ax_tc.tick_params(axis="y", labelcolor="r")
    ax_tc.grid(alpha=0.3)

    ax_c = ax_tc.twinx()
    ax_c.plot(t_min, C_win, "b-", lw=1.6, label="浓度 C")
    ax_c.set_ylabel("浓度 C", color="b")
    ax_c.tick_params(axis="y", labelcolor="b")

    ax_tc.axvspan(t_min[0], split_min, color="0.88", alpha=0.7)
    ax_tc.axvline(split_min, color="k", ls=":", lw=1.2)
    ax_tc.text(split_min, ax_tc.get_ylim()[1], "  输入|输出",
               va="top", ha="left", fontsize=9)

    t0_min, t1_min = t_min[0], t_min[-1]
    in_end_min = t_abs[n_in - 1] / 60.0
    ax_tc.set_title(
        f"温度 / 浓度演化  t={t0_min:.1f}~{t1_min:.1f}min\n"
        f"输入段 {t0_min:.1f}~{in_end_min:.1f}min  |  "
        f"输出段 {split_min:.1f}~{t1_min:.1f}min")

    lines1, labels1 = ax_tc.get_legend_handles_labels()
    lines2, labels2 = ax_c.get_legend_handles_labels()
    ax_tc.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle(
        f"窗口起点 t0={t0}pt ({t_abs[0]:.0f}s)  |  "
        f"窗口总长 {len(t_abs)} 点 = {t_min[-1]-t_min[0]:.1f}min",
        fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview(case_name: str, starts, selected_idx, dt, n_in, n_out,
                  save_path: str) -> None:
    """总览图：5 个切片在整条工况时间轴上的位置。"""
    n_time = starts[-1] + n_in + n_out
    t_end = n_time * dt / 60.0

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, t_end)
    ax.set_ylim(0, len(selected_idx) + 1)
    ax.set_xlabel("时间 (min)")
    ax.set_yticks(range(1, len(selected_idx) + 1))
    ax.set_yticklabels([f"切片#{i}" for i in selected_idx])
    ax.set_title(f"切片位置总览 [{case_name}]  （灰=输入15min，蓝=输出5min）")

    split_min = n_in * dt / 60.0
    win_min = (n_in + n_out) * dt / 60.0
    for row, (si, t0) in enumerate(zip(selected_idx, starts), start=1):
        t_start = t0 * dt / 60.0
        ax.barh(row, split_min, left=t_start, height=0.6,
                color="0.75", edgecolor="k", linewidth=0.5)
        ax.barh(row, win_min - split_min, left=t_start + split_min, height=0.6,
                color="steelblue", alpha=0.7, edgecolor="k", linewidth=0.5)
        ax.text(t_start + win_min + 1, row, f"#{si}", va="center", fontsize=9)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    cfg = get_default_config()
    win = cfg.window
    ap = argparse.ArgumentParser(description="可视化滑动窗口切片数据")
    ap.add_argument("--case", default=None, help="工况名，默认取第一个工况")
    ap.add_argument("--npz", default=cfg.path.npz_path)
    ap.add_argument("--n_show", type=int, default=5, help="展示切片数（含首尾）")
    ap.add_argument("--out", default=os.path.join(cfg.path.results_dir, "slice_inspect"),
                    help="输出目录")
    ap.add_argument("--l_step", type=int, default=8,
                    help="3D 图粒径下采样步长（仅影响绘图密度）")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    names = [str(x) for x in data["case_names"]]
    case = args.case or names[0]
    if case not in names:
        raise ValueError(f"工况 {case!r} 不存在。可选: {names}")

    ci = names.index(case)
    T = data["T"][ci]
    C = data["C"][ci]
    psd = data["psd"][ci]
    L = data["L"]
    dt = float(data["dt"])
    n_time = T.shape[0]

    starts = window_starts(n_time, win.n_window, win.window_stride_pts)
    pick_idx = pick_slice_indices(args.n_show, len(starts))
    picked_starts = [starts[i] for i in pick_idx]

    out_dir = os.path.join(args.out, case)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[SliceViz] 工况: {case}")
    print(f"  数据: n_time={n_time}  dt={dt}s  n_L={L.size}")
    print(f"  窗口: 输入={win.in_seconds}s({win.n_in}pt) + "
          f"输出={win.out_seconds}s({win.n_out}pt)  步长={win.window_stride_pts}pt")
    print(f"  总切片数: {len(starts)}  本次展示: {len(pick_idx)} 个 "
          f"(索引 {pick_idx})")

    plot_overview(case, starts, pick_idx, dt, win.n_in, win.n_out,
                  os.path.join(out_dir, "slice_overview.png"))

    for si, t0 in zip(pick_idx, picked_starts):
        t_abs = (t0 + np.arange(win.n_window)) * dt
        psd_win = psd[t0:t0 + win.n_window, :]
        T_win = T[t0:t0 + win.n_window]
        C_win = C[t0:t0 + win.n_window]

        in_start = t_abs[0] / 60.0
        in_end = t_abs[win.n_in - 1] / 60.0
        out_start = t_abs[win.n_in] / 60.0
        out_end = t_abs[-1] / 60.0
        print(f"  切片#{si:3d}  t0={t0:4d}pt  "
              f"输入 {in_start:6.1f}~{in_end:6.1f}min  "
              f"输出 {out_start:6.1f}~{out_end:6.1f}min  "
              f"PSD峰值={psd_win.max():.3e}")

        plot_slice_panel(
            case, si, t0, t_abs, L, psd_win, T_win, C_win,
            n_in=win.n_in,
            save_path=os.path.join(out_dir, f"slice_{si:03d}.png"),
            l_step=args.l_step,
        )

    print(f"[OK] 切片可视化已保存到 {out_dir}")
    print("     - slice_overview.png  （5 个切片在全程时间轴上的位置）")
    print("     - slice_XXX.png       （各切片的 PSD 3D 曲面 + 温度/浓度曲线）")


if __name__ == "__main__":
    main()
