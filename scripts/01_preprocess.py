"""
脚本 01：原始 .mat → 抽稀 .npz。

输出 ``dataset_3s.npz`` 仅含 T/C/PSD 时间序列；Branch 中的 T_future 在
``PBEWindowData`` 切窗时从 T 序列截取，**改 Branch 结构后无需重跑本脚本**。

用法::
    python scripts/01_preprocess.py
    python scripts/01_preprocess.py --raw_mat data/Simulation_Data_DeepONet.mat --stride 3
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from donpbe.config import get_default_config
from donpbe.preprocess import MatToNpzConverter


def main():
    cfg = get_default_config()
    ap = argparse.ArgumentParser(description="mat -> npz 预处理")
    ap.add_argument("--raw_mat", default=cfg.path.raw_mat)
    ap.add_argument("--npz", default=cfg.path.npz_path)
    ap.add_argument("--stride", type=int, default=cfg.window.raw_stride)
    args = ap.parse_args()

    conv = MatToNpzConverter(
        raw_mat=args.raw_mat, npz_path=args.npz, raw_stride=args.stride)
    conv.run()
    print("[OK] 预处理完成。")


if __name__ == "__main__":
    main()
