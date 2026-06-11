"""
工具模块：评估指标与可视化。
"""

import glob
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 配置中文字体（Windows 自带 Microsoft YaHei / SimHei），避免中文显示为方块
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def resolve_run_dir(run_arg: str | None, results_dir: str) -> str:
    """解析训练 run 目录。

    接受：
      - ``None`` → ``results_dir`` 下最新含 ``weights/best.pt`` 的目录
      - run 目录 ``results/20260611_134539``
      - 权重文件 ``.../weights/best.pt`` 或 ``.../best.pt``
    """
    if not run_arg:
        pat = os.path.join(results_dir, "*", "weights", "best.pt")
        runs = glob.glob(pat)
        if not runs:
            raise FileNotFoundError(
                f"{results_dir} 下没有可用的训练结果。")
        return os.path.dirname(os.path.dirname(max(runs, key=os.path.getmtime)))

    p = os.path.abspath(run_arg)
    if os.path.isfile(p):
        parent = os.path.dirname(p)
        if os.path.basename(parent) == "weights":
            return os.path.dirname(parent)
        raise FileNotFoundError(
            f"权重路径须为 .../weights/best.pt，当前: {p}")

    if os.path.isdir(p):
        w = os.path.join(p, "weights", "best.pt")
        if os.path.isfile(w):
            return p
        if os.path.basename(p) == "weights" and os.path.isfile(
                os.path.join(p, "best.pt")):
            return os.path.dirname(p)

    raise FileNotFoundError(
        f"未找到训练 run 或权重: {run_arg}\n"
        f"请传 run 目录（含 weights/best.pt），例如 results/20260611_134539")


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """相对 L2 误差。"""
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def plot_loss_curve(train_hist, val_hist, save_path: str) -> None:
    """绘制训练/验证损失曲线（对数纵轴）。"""
    plt.figure(figsize=(8, 5))
    epochs = np.arange(1, len(train_hist) + 1)
    plt.semilogy(epochs, train_hist, label="train")
    if val_hist is not None and len(val_hist):
        plt.semilogy(epochs, val_hist, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE, normalized)")
    plt.title("DeepONet 训练损失曲线")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_psd_compare(L_eval, true_grid, pred_grid, tau_seconds,
                     case_name: str, save_path: str, ncols: int = 5) -> None:
    """绘制后 5min 若干时刻的 PSD 预测 vs 真值。

    Parameters
    ----------
    true_grid, pred_grid : (n_out, n_L_eval)  真实尺度的 PSD
    tau_seconds : (n_out,) 预测窗口内相对时间
    """
    n_out = true_grid.shape[0]
    sel = np.linspace(0, n_out - 1, min(ncols * 2, n_out), dtype=int)
    nrows = int(np.ceil(len(sel) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows),
                             squeeze=False)
    for k, ti in enumerate(sel):
        ax = axes[k // ncols][k % ncols]
        ax.plot(L_eval, true_grid[ti], "b-", lw=1.3, label="仿真")
        ax.plot(L_eval, pred_grid[ti], "r--", lw=1.3, label="预测")
        ax.set_title(f"τ={tau_seconds[ti]:.0f}s", fontsize=9)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    for k in range(len(sel), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"后5min PSD 预测 vs 仿真 [{case_name}]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_concentration(tau_seconds, true_c, pred_c, case_name: str,
                       save_path: str) -> None:
    """绘制后 5min 浓度预测 vs 真值。"""
    plt.figure(figsize=(8, 5))
    plt.plot(tau_seconds, true_c, "b-", lw=1.6, label="仿真")
    plt.plot(tau_seconds, pred_c, "r--", lw=1.6, label="预测")
    plt.xlabel("τ (s)")
    plt.ylabel("浓度 C")
    plt.title(f"后5min 浓度预测 vs 仿真 [{case_name}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_spacetime(L_eval, tau_seconds, true_grid, pred_grid,
                   case_name: str, save_path: str) -> None:
    """绘制 (L, τ) 时空热力图：真值 / 预测 / 误差。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    titles = ["仿真", "预测", "误差(预测-仿真)"]
    grids = [true_grid.T, pred_grid.T, (pred_grid - true_grid).T]
    extent = [tau_seconds[0], tau_seconds[-1], L_eval[0], L_eval[-1]]
    for ax, title, grid in zip(axes, titles, grids):
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       extent=extent, cmap="viridis")
        ax.set_xlabel("τ (s)")
        ax.set_ylabel("L (μm)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"PSD 时空演化 [{case_name}]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
