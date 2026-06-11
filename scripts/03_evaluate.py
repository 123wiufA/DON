"""
脚本 03：评估已训练模型并可视化后 5min PSD 预测。

用法::
    python scripts/03_evaluate.py                       # 自动用最新结果目录
    python scripts/03_evaluate.py --run results/20260610_xxxxxx
    python scripts/03_evaluate.py --case CR_1_14
"""

import os
import sys
import glob
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from donpbe.config import get_default_config, apply_run_config
from donpbe.device import setup_device
from donpbe.dataset import PBEWindowData
from donpbe.model import DeepONet
from donpbe.utils import (relative_l2, plot_psd_compare, plot_spacetime,
                          plot_concentration)


def latest_run(results_dir: str) -> str:
    runs = [d for d in glob.glob(os.path.join(results_dir, "*"))
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "weights", "best.pt"))]
    if not runs:
        raise FileNotFoundError(f"{results_dir} 下没有可用的训练结果。")
    return max(runs, key=os.path.getmtime)


def main():
    cfg = get_default_config()
    ap = argparse.ArgumentParser(description="评估 DeepONet")
    ap.add_argument("--run", default=None, help="结果目录，默认取最新")
    ap.add_argument("--npz", default=cfg.path.npz_path)
    ap.add_argument("--case", default=None, help="指定可视化的工况名")
    args = ap.parse_args()

    device = setup_device()
    run_dir = args.run or latest_run(cfg.path.results_dir)
    print(f"[Eval] 结果目录: {run_dir}")
    cfg = apply_run_config(cfg, run_dir)

    data = PBEWindowData(args.npz, cfg)
    data.summary()

    model = DeepONet(
        branch_dim=cfg.branch_dim, trunk_dim=2,
        branch_hiddens=cfg.model.branch_hiddens,
        trunk_hiddens=cfg.model.trunk_hiddens,
        latent_dim=cfg.model.latent_dim,
        activation=cfg.model.activation,
        n_out=cfg.window.n_out,
        conc_trunk_hiddens=cfg.model.conc_trunk_hiddens).to(device)
    model.load_state_dict(torch.load(
        os.path.join(run_dir, "weights", "best.pt"),
        map_location=device, weights_only=True))
    model.eval()

    trunk = torch.from_numpy(data.trunk_grid).to(device)
    trunk_conc = torch.from_numpy(data.trunk_conc_grid).to(device)
    nrm = data.normalizer
    n_out, n_L_eval = cfg.window.n_out, cfg.window.n_L_eval
    tau_seconds = np.arange(n_out) * data.dt

    # ---- holdout（神经网络从未接触）整体误差 ----
    b_ho, p_ho, c_ho = data.build_split("holdout")
    if b_ho.shape[0] == 0:
        print("[Eval] 无 holdout 样本（n_holdout_cases=0）。"); return
    with torch.no_grad():
        pred_psd, pred_conc = model(
            torch.from_numpy(b_ho).to(device), trunk, trunk_conc)
        pred_psd = pred_psd.cpu().numpy()
        pred_conc = pred_conc.cpu().numpy()
    mse_psd = float(np.mean((np.maximum(pred_psd, 0) - p_ho) ** 2))
    rl2_psd = relative_l2(np.maximum(pred_psd, 0), p_ho)
    rl2_conc = relative_l2(pred_conc, c_ho)
    print(f"[Eval] holdout 样本={b_ho.shape[0]}  "
          f"PSD: MSE(norm)={mse_psd:.4e} Rel.L2={rl2_psd:.4e}  |  "
          f"浓度 Rel.L2={rl2_conc:.4e}")

    # ---- 可视化一个 holdout 窗口（PSD + 浓度）----
    case_name = args.case or (data.case_names[data.holdout_cases[0]]
                              if data.holdout_cases else data.case_names[0])
    ci = data.case_names.index(case_name)
    t0 = data._window_starts()[0]
    branch_vec = data._build_branch(ci, t0)[None, :]
    with torch.no_grad():
        pp, cc = model(torch.from_numpy(branch_vec).to(device), trunk, trunk_conc)
        pp = pp.cpu().numpy()[0]
        cc = cc.cpu().numpy()[0]
    pred_grid = nrm.denorm_n(np.maximum(pp, 0)).reshape(n_L_eval, n_out).T
    true_grid = nrm.denorm_n(data._build_label_psd(ci, t0)).reshape(n_L_eval, n_out).T
    pred_c = nrm.C_min + cc * (nrm.C_max - nrm.C_min)
    true_c = nrm.C_min + data._build_label_conc(ci, t0) * (nrm.C_max - nrm.C_min)

    out_dir = os.path.join(run_dir, "eval")
    plot_psd_compare(data.L_eval, true_grid, pred_grid, tau_seconds,
                     case_name, os.path.join(out_dir, f"psd_{case_name}.png"))
    plot_spacetime(data.L_eval, tau_seconds, true_grid, pred_grid,
                   case_name, os.path.join(out_dir, f"spacetime_{case_name}.png"))
    plot_concentration(tau_seconds, true_c, pred_c, case_name,
                       os.path.join(out_dir, f"conc_{case_name}.png"))
    print(f"[OK] 评估图（PSD+浓度）已保存到 {out_dir}")


if __name__ == "__main__":
    main()
