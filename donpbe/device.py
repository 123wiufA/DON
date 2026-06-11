"""
设备与随机种子配置模块。

针对 RTX 4060 (Ada, CUDA) + i9-13900H 做了如下优化：
  - 自动选择 CUDA，打印显卡信息
  - 开启 cudnn.benchmark 以加速固定尺寸卷积/矩阵运算
  - 允许 TF32（Ada 架构支持，矩阵乘更快）
  - 统一设置随机种子，保证可复现
"""

import os
import random
import numpy as np
import torch


def setup_device(verbose: bool = True) -> torch.device:
    """选择并返回训练设备（优先 CUDA）。"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        # Ada 架构支持 TF32，提升 matmul 吞吐
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if verbose:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[Device] 使用 GPU: {name} ({total:.1f} GB), "
                  f"CUDA {torch.version.cuda}, torch {torch.__version__}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[Device] 未检测到 CUDA，使用 CPU。"
                  "若有 RTX 4060，请安装 CUDA 版 PyTorch（见 README）。")
    return device


def set_seed(seed: int = 42) -> None:
    """统一设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
