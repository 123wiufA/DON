"""
donpbe —— 基于 DeepONet 的 PBE 结晶过程时序算子学习库（多任务）。

任务：用前 15min 的过程信息（温度、浓度、粒度分布演化），同时预测后 5min 的
      粒度分布 PSD `n(L, τ)` 与浓度 `C(τ)`。窗口按 5min 步长滚动切分，
      工况随机划分为 训练/测试/推理验证(holdout) 三部分。

模块:
    config      全局配置（dataclass）
    device      设备与随机种子（适配 RTX 4060 / CUDA）
    preprocess  原始 .mat → 抽稀 .npz
    dataset     滚动窗口切分 + Branch/Trunk + 浓度标签 + 随机划分 + 归一化
    model       DeepONet 双算子（PSD + 浓度，PyTorch，多任务）
    trainer     训练器（联合损失 / AMP / 调度 / 检查点）
    utils       指标与可视化（PSD + 浓度）
"""

from .config import Config, get_default_config
from .device import setup_device, set_seed
from .preprocess import MatToNpzConverter
from .dataset import PBEWindowData, Normalizer
from .model import DeepONet
from .trainer import Trainer
from .predictor import Predictor

__all__ = [
    "Config", "get_default_config",
    "setup_device", "set_seed",
    "MatToNpzConverter",
    "PBEWindowData", "Normalizer",
    "DeepONet", "Trainer", "Predictor",
]
