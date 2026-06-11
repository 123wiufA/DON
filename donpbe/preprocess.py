"""
预处理模块：把原始 MATLAB .mat 数据集转换为读取极快的 .npz。

支持：
  - MATLAB v7.3（HDF5，推荐）—— h5py
  - MATLAB v7 及以下（经典格式）—— scipy.io.loadmat

原始数据结构：
  Dataset/CR_x_xx/{Time_s, Temp_K, Conc, L_mid_um, psd, snapshot_times, C0, n_L0}
"""

import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

_HDF5_SIG = b"\x89HDF\r\n\x1a\n"

REQUIRED_KEYS = ["Time_s", "Temp_K", "Conc", "L_mid_um", "psd",
                 "snapshot_times", "C0", "n_L0"]


def inspect_mat_file(path: str) -> Tuple[int, str]:
    """检查 .mat 文件头，返回 (size_bytes, 诊断信息)。不加载全量数据。

    v7.3 的 HDF5 超块通常在偏移 512 处，必须扫描 >=520 字节才能可靠识别。
    """
    size = os.path.getsize(path)
    probe = min(size, 8192)
    with open(path, "rb") as fp:
        head = fp.read(probe)

    head128 = head[:128]
    has_hdf5 = _HDF5_SIG in head
    has_matlab_txt = b"MATLAB" in head128
    all_zero_head = head128 == b"\x00" * 128

    if all_zero_head and not has_hdf5:
        # 再扫文件尾部（部分损坏文件 HDF5 签名偏后）
        with open(path, "rb") as fp:
            fp.seek(max(0, size - 8192))
            tail = fp.read()
        has_hdf5 = _HDF5_SIG in tail
        if not has_hdf5:
            return size, (
                "文件头 128 字节全为 0，且未找到 HDF5 签名 —— "
                "这不是合法的 .mat，多为 save 中断/拷贝不完整导致。")

    if has_hdf5:
        return size, "OK (MATLAB v7.3 / HDF5)"
    if has_matlab_txt:
        return size, "OK (MATLAB v7, scipy)"
    return size, (
        "未识别为 MATLAB .mat（无 MATLAB 文件头且无 HDF5 签名）。")


def assert_mat_readable(path: str) -> Tuple[int, str]:
    """校验文件可读；返回 (size_bytes, format)，format 为 'hdf5' 或 'v7'。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到原始数据: {path}")
    size, msg = inspect_mat_file(path)
    if not msg.startswith("OK"):
        raise ValueError(f"{path}\n  → {msg}")
    fmt = "hdf5" if "HDF5" in msg else "v7"
    return size, fmt


# 向后兼容旧名
assert_mat_v73 = assert_mat_readable


@dataclass
class MatToNpzConverter:
    """把 .mat 转成抽稀后的 .npz。"""

    raw_mat: str
    npz_path: str
    raw_stride: int = 3

    def _log(self, msg: str, verbose: bool):
        if verbose:
            print(msg, flush=True)

    @staticmethod
    def _read_psd_strided(psd_raw, t_idx: np.ndarray,
                          n_time_full: int) -> np.ndarray:
        arr = np.asarray(psd_raw)
        shp = arr.shape
        if shp[0] == n_time_full:              # (N_time, N_L)
            return arr[t_idx, :].astype(np.float32)
        if shp[1] == n_time_full:              # (N_L, N_time)
            return arr[:, t_idx].astype(np.float32).T
        raise ValueError(
            f"psd 形状 {shp} 与时间点数 {n_time_full} 不匹配。"
            f"若 psd 仅为 snapshot 维 (N_L×N_snap)，需先在 MATLAB 插值到 1s 网格。")

    @staticmethod
    def _scipy_case_names(root) -> List[str]:
        if hasattr(root, "_fieldnames"):
            return sorted(n for n in root._fieldnames if not n.startswith("_"))
        if isinstance(root, np.ndarray) and root.dtype.names:
            return sorted(n for n in root.dtype.names if not n.startswith("_"))
        raise ValueError("Dataset 不是预期的 MATLAB struct（无字段名）。")

    @staticmethod
    def _scipy_get_group(root, name: str):
        if hasattr(root, name):
            return getattr(root, name)
        if isinstance(root, np.ndarray) and root.dtype.names and name in root.dtype.names:
            item = root[name]
            return item[0, 0] if isinstance(item, np.ndarray) and item.size == 1 else item
        raise KeyError(name)

    @staticmethod
    def _scipy_get_field(group, key: str) -> np.ndarray:
        if hasattr(group, key):
            return np.asarray(getattr(group, key))
        if isinstance(group, np.ndarray) and group.dtype.names and key in group.dtype.names:
            item = group[key]
            return np.asarray(item[0, 0] if isinstance(item, np.ndarray) and item.size == 1 else item)
        raise KeyError(key)

    def _fill_arrays_hdf5(self, f, verbose: bool):
        import h5py

        self._log("[Preprocess] HDF5 已打开，扫描工况...", verbose)
        root = f["Dataset"] if "Dataset" in f else f
        names: List[str] = [k for k in sorted(root.keys())
                            if isinstance(root[k], h5py.Group)]
        if not names:
            raise ValueError("文件中没有工况组（Dataset/CR_xx）。")

        g0 = root[names[0]]
        for k in REQUIRED_KEYS:
            if k not in g0:
                raise ValueError(f"工况 {names[0]} 缺少字段 {k}")

        return self._fill_arrays_from_cases(
            names,
            lambda nm: root[nm],
            lambda g, k: np.asarray(g[k]),
            verbose,
        )

    def _fill_arrays_scipy(self, root, verbose: bool):
        names = self._scipy_case_names(root)
        if not names:
            raise ValueError("Dataset 中没有工况。")

        g0 = self._scipy_get_group(root, names[0])
        for k in REQUIRED_KEYS:
            if not (hasattr(g0, k) or (
                    isinstance(g0, np.ndarray) and g0.dtype.names and k in g0.dtype.names)):
                raise ValueError(f"工况 {names[0]} 缺少字段 {k}")

        return self._fill_arrays_from_cases(
            names,
            lambda nm: self._scipy_get_group(root, nm),
            self._scipy_get_field,
            verbose,
        )

    def _fill_arrays_from_cases(self, names, get_group, get_field, verbose: bool):
        g0 = get_group(names[0])
        self._log(f"[Preprocess] 读取网格元数据 ({names[0]})...", verbose)

        n_time_full = int(np.asarray(get_field(g0, "Time_s")).size)
        t_idx = np.arange(0, n_time_full, self.raw_stride)
        L = np.asarray(get_field(g0, "L_mid_um")).ravel().astype(np.float32)
        n_L = L.size
        t_full = np.asarray(get_field(g0, "Time_s")).ravel().astype(np.float64)
        t = t_full[t_idx].astype(np.float32)
        dt = float(t[1] - t[0]) if t.size > 1 else float(self.raw_stride)

        n_cases = len(names)
        n_time = t_idx.size
        self._log(
            f"[Preprocess] 工况数={n_cases}, 原始时间点={n_time_full} "
            f"-> 抽稀(stride={self.raw_stride})后={n_time}, "
            f"L={n_L}, dt={dt}s", verbose)

        T = np.zeros((n_cases, n_time), dtype=np.float32)
        C = np.zeros((n_cases, n_time), dtype=np.float32)
        psd = np.zeros((n_cases, n_time, n_L), dtype=np.float32)

        for i, nm in enumerate(names):
            self._log(f"  [{i+1}/{n_cases}] 读取 {nm} ...", verbose)
            g = get_group(nm)
            T[i] = np.asarray(get_field(g, "Temp_K")).ravel()[t_idx].astype(np.float32)
            C[i] = np.asarray(get_field(g, "Conc")).ravel()[t_idx].astype(np.float32)
            psd[i] = self._read_psd_strided(get_field(g, "psd"), t_idx, n_time_full)
            if verbose:
                print(f"       psd_max={psd[i].max():.3e}", flush=True)

        return names, T, C, psd, L, t, dt

    def _save_npz(self, names, T, C, psd, L, t, dt, verbose: bool) -> str:
        self._log("[Preprocess] 写入 .npz ...", verbose)
        os.makedirs(os.path.dirname(self.npz_path) or ".", exist_ok=True)
        np.savez(
            self.npz_path,
            T=T, C=C, psd=psd, L=L, t=t,
            case_names=np.array(names),
            dt=np.float32(dt),
            raw_stride=np.int64(self.raw_stride),
        )
        size_mb = os.path.getsize(self.npz_path) / 1e6
        self._log(f"[Preprocess] 已保存: {self.npz_path} ({size_mb:.1f} MB)", verbose)
        return self.npz_path

    def run(self, verbose: bool = True) -> str:
        """执行转换，返回输出 .npz 路径。"""
        size_b, fmt = assert_mat_readable(self.raw_mat)
        self._log(
            f"[Preprocess] 格式={fmt}, 大小 {size_b / 1e6:.1f} MB", verbose)

        if fmt == "hdf5":
            import h5py

            self._log("[Preprocess] 打开 HDF5（大文件可能需 10~60s）...", verbose)
            with h5py.File(self.raw_mat, "r") as f:
                names, T, C, psd, L, t, dt = self._fill_arrays_hdf5(f, verbose)
        else:
            import scipy.io as sio

            self._log(
                "[Preprocess] scipy.loadmat 读取 v7（约 1GB 时可能需 1~3 min）...",
                verbose)
            t0 = time.time()
            try:
                mat = sio.loadmat(
                    self.raw_mat, struct_as_record=False, squeeze_me=True)
            except NotImplementedError as exc:
                if "v7.3" not in str(exc):
                    raise
                self._log(
                    "[Preprocess] 实为 v7.3，改用 h5py ...", verbose)
                import h5py

                with h5py.File(self.raw_mat, "r") as f:
                    names, T, C, psd, L, t, dt = self._fill_arrays_hdf5(
                        f, verbose)
                return self._save_npz(names, T, C, psd, L, t, dt, verbose)

            self._log(f"[Preprocess] loadmat 完成 ({time.time() - t0:.1f}s)", verbose)
            if "Dataset" not in mat:
                raise ValueError("文件中缺少顶层变量 Dataset。")
            names, T, C, psd, L, t, dt = self._fill_arrays_scipy(
                mat["Dataset"], verbose)

        return self._save_npz(names, T, C, psd, L, t, dt, verbose)
