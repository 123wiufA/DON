"""
快速检查 Simulation_Data_DeepONet.mat 是否可被预处理读取（不加载全量 psd）。

用法::
    python scripts/00_check_mat.py
    python scripts/00_check_mat.py --raw_mat path/to/file.mat
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from donpbe.config import get_default_config
from donpbe.preprocess import assert_mat_readable, inspect_mat_file, REQUIRED_KEYS


def _check_hdf5(path: str):
    import h5py

    t0 = time.time()
    print("打开 HDF5...", flush=True)
    with h5py.File(path, "r") as f:
        print(f"  打开耗时 {time.time() - t0:.1f}s")
        root = f["Dataset"] if "Dataset" in f else f
        names = [k for k in sorted(root.keys()) if hasattr(root[k], "keys")]
        print(f"工况数: {len(names)}  示例: {names[:3]}")
        g = root[names[0]]
        missing = [k for k in REQUIRED_KEYS if k not in g]
        if missing:
            print(f"[FAIL] {names[0]} 缺少字段: {missing}")
            sys.exit(3)
        for k in ["Time_s", "Temp_K", "Conc", "L_mid_um", "psd"]:
            ds = g[k]
            print(f"  {k}: shape={ds.shape}, dtype={ds.dtype}")
        t1 = time.time()
        _ = g["Time_s"][0:3]
        _ = g["psd"][0:2, 0:5]
        print(f"  试读小切片耗时 {time.time() - t1:.2f}s")


def _check_scipy(path: str):
    import scipy.io as sio
    import numpy as np

    print("scipy.loadmat（仅读结构，可能较慢）...", flush=True)
    t0 = time.time()
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    print(f"  loadmat 耗时 {time.time() - t0:.1f}s")
    if "Dataset" not in mat:
        print("[FAIL] 缺少顶层 Dataset")
        sys.exit(3)
    root = mat["Dataset"]
    if hasattr(root, "_fieldnames"):
        names = sorted(n for n in root._fieldnames if not n.startswith("_"))
    else:
        names = sorted(n for n in root.dtype.names if not n.startswith("_"))
    print(f"工况数: {len(names)}  示例: {names[:3]}")
    g = getattr(root, names[0])
    for k in REQUIRED_KEYS:
        if not hasattr(g, k):
            print(f"[FAIL] {names[0]} 缺少字段 {k}")
            sys.exit(3)
    for k in ["Time_s", "Temp_K", "Conc", "L_mid_um", "psd"]:
        arr = np.asarray(getattr(g, k))
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")


def main():
    cfg = get_default_config()
    ap = argparse.ArgumentParser(description="检查 .mat 文件头与结构")
    ap.add_argument("--raw_mat", default=cfg.path.raw_mat)
    args = ap.parse_args()

    path = args.raw_mat
    print(f"文件: {path}")
    if not os.path.isfile(path):
        print("[FAIL] 文件不存在")
        sys.exit(1)

    size, msg = inspect_mat_file(path)
    print(f"大小: {size / 1e6:.1f} MB")
    print(f"文件头: {msg}")
    if not msg.startswith("OK"):
        print("\n[FAIL] 无法预处理，请按提示修复后重试。")
        sys.exit(2)

    _, fmt = assert_mat_readable(path)
    if fmt == "hdf5":
        _check_hdf5(path)
    else:
        _check_scipy(path)

    print("\n[OK] 结构正常，可运行: python scripts/01_preprocess.py")
    print(f"     --raw_mat \"{path}\"")


if __name__ == "__main__":
    main()
