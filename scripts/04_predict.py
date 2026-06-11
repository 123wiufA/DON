"""
脚本 04：自定义起始时刻的单窗口预测。

指定某工况、某个起始时刻（分钟），取「该时刻起前 15min」作为输入，
预测随后 5min 的 PSD 与浓度，并与仿真真值对比出图。

用法::
    # 在工况 CR_1_13 上，从第 40 分钟开始取 15min 输入，预测 55~60min
    python scripts/04_predict.py --case CR_1_13 --start_min 40

    # 指定结果目录与起始秒
    python scripts/04_predict.py --case CR_1_08 --start_sec 2400 --run results/2026xxxx_xxxxxx
"""

import os
import sys
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from donpbe.config import get_default_config
from donpbe.device import setup_device
from donpbe.predictor import Predictor
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
    ap = argparse.ArgumentParser(description="自定义起始时刻的单窗口预测")
    ap.add_argument("--case", required=True, help="工况名，如 CR_1_13")
    ap.add_argument("--start_min", type=float, default=None,
                    help="输入窗口起始时刻（分钟）")
    ap.add_argument("--start_sec", type=float, default=None,
                    help="输入窗口起始时刻（秒），与 --start_min 二选一")
    ap.add_argument("--run", default=None, help="结果目录，默认取最新")
    ap.add_argument("--npz", default=cfg.path.npz_path)
    args = ap.parse_args()

    if args.start_min is None and args.start_sec is None:
        args.start_min = 0.0
    start_sec = (args.start_sec if args.start_sec is not None
                 else args.start_min * 60.0)

    device = setup_device()
    run_dir = args.run or latest_run(cfg.path.results_dir)
    print(f"[Predict] 结果目录: {run_dir}")

    predictor = Predictor(run_dir, args.npz, cfg, device)
    res = predictor.predict_window(args.case, start_sec)

    ir, orng = res["input_range"], res["output_range"]
    print(f"[Predict] 工况 {args.case}")
    print(f"  输入段(15min): {ir[0]:.0f} ~ {ir[1]:.0f}s "
          f"({ir[0]/60:.1f} ~ {ir[1]/60:.1f}min)")
    print(f"  预测段(5min) : {orng[0]:.0f} ~ {orng[1]:.0f}s "
          f"({orng[0]/60:.1f} ~ {orng[1]/60:.1f}min)")

    if res["true_psd"] is not None:
        rl2_psd = relative_l2(res["pred_psd"], res["true_psd"])
        rl2_c = relative_l2(res["pred_conc"], res["true_conc"])
        print(f"  PSD Rel.L2 = {rl2_psd:.4e}   浓度 Rel.L2 = {rl2_c:.4e}")
    else:
        print("  (该预测段超出数据范围，无真值对比)")

    out_dir = os.path.join(run_dir, "predict", f"{args.case}_t{int(start_sec)}")
    tag = f"{args.case}_t{int(start_sec)}"
    if res["true_psd"] is not None:
        plot_psd_compare(res["L_eval"], res["true_psd"], res["pred_psd"],
                         res["tau_seconds"], tag,
                         os.path.join(out_dir, f"psd_{tag}.png"))
        plot_spacetime(res["L_eval"], res["tau_seconds"], res["true_psd"],
                       res["pred_psd"], tag,
                       os.path.join(out_dir, f"spacetime_{tag}.png"))
        plot_concentration(res["tau_seconds"], res["true_conc"], res["pred_conc"],
                           tag, os.path.join(out_dir, f"conc_{tag}.png"))
    else:
        # 无真值时仅画预测
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(res["tau_seconds"], res["pred_conc"], "r--", label="预测浓度")
        plt.xlabel("τ (s)"); plt.ylabel("浓度 C"); plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(os.path.join(out_dir, f"conc_{tag}.png"), dpi=150)
        plt.close()
    print(f"[OK] 预测图已保存到 {out_dir}")


if __name__ == "__main__":
    main()
