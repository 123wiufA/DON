"""
全局配置模块。

用 dataclass 把项目所有可调参数集中管理，分为四组：
  - PathConfig    路径配置（原始 .mat、预处理 .npz、结果目录）
  - WindowConfig  时间窗口与采样配置（前 15min 预测后 5min）
  - ModelConfig   DeepONet 网络结构
  - TrainConfig   训练超参数

修改实验设置时，优先改这里，避免散落到各脚本中。
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os


# 项目根目录（donpbe 的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PathConfig:
    """路径配置。"""

    # 原始 MATLAB v7.3 数据（15 工况，dt=1s，10801 点）
    raw_mat: str = os.path.join(
        PROJECT_ROOT, "data", "Simulation_Data_DeepONet.mat")
    # 预处理输出的快速读取数据集
    npz_path: str = os.path.join(PROJECT_ROOT, "data", "dataset_3s.npz")
    # 结果目录（权重、归一化参数、图、日志）
    results_dir: str = os.path.join(PROJECT_ROOT, "results")


@dataclass
class WindowConfig:
    """时间窗口与采样配置。

    原始数据 dt=1s、10801 点（0~10800s）。预处理按 ``raw_stride`` 抽稀，
    得到 dt = raw_stride 秒的序列；随后用滑动窗口切成
    「输入段(前15min) + 输出段(后5min)」的样本。
    """

    raw_stride: int = 3            # 原始序列抽稀步长：每 3 点取 1 → dt=3s
    dt: float = 3.0                # 抽稀后的时间步长（秒），= raw_stride

    in_seconds: int = 900          # 输入窗口时长：15 min
    out_seconds: int = 300         # 输出窗口时长：5 min
    window_stride_pts: int = 100   # 滑动窗口步长（抽稀后点数，100 点 = 300s = 5min）
                                   # 即 0-15→15-20, 5-20→20-25, ... 逐 5min 滚动

    n_T_sensors: int = 300         # Branch：输入段温度（= n_in，全分辨率 15min@3s）
    n_C_sensors: int = 300         # Branch：输入段浓度（= n_in）
    n_L_sensors: int = 128         # Branch：PSD 历史 n(L,t) 每时刻 L 向采样点数
    n_T_future_sensors: int = 100  # Branch：输出段计划温度（= n_out，全分辨率 5min@3s）
    n_L_eval: int = 200            # Trunk：输出 PSD 的粒径评估点数

    @property
    def n_in(self) -> int:
        """输入窗口点数。"""
        return int(round(self.in_seconds / self.dt))

    @property
    def n_out(self) -> int:
        """输出窗口点数。"""
        return int(round(self.out_seconds / self.dt))

    @property
    def n_window(self) -> int:
        """单个样本窗口总点数。"""
        return self.n_in + self.n_out


@dataclass
class ModelConfig:
    """DeepONet 网络结构配置。"""

    branch_hiddens: List[int] = field(default_factory=lambda: [256, 512, 512, 256])
    trunk_hiddens: List[int] = field(default_factory=lambda: [128, 256, 256, 128])
    latent_dim: int = 128          # Branch / Trunk 公共输出维度 p
    conc_trunk_hiddens: List[int] = field(default_factory=lambda: [128, 128, 128])  # 浓度 Trunk
    activation: str = "tanh"


@dataclass
class TrainConfig:
    """训练超参数。"""

    # 工况随机划分（共 15 条）：
    #   训练集    n_train_cases 条  —— 参与反向传播
    #   测试集    n_val_cases   条  —— 训练时监控 val_loss（不反传）
    #   推理验证  n_holdout_cases 条 —— 完全不接触，仅训练后推理评估
    n_train_cases: int = 11
    n_val_cases: int = 3
    n_holdout_cases: int = 1
    split_seed: int = 42           # 工况随机划分种子
    # 如需固定指定（覆盖随机划分），填工况名元组；否则留 None
    train_cases: Tuple[str, ...] = None
    val_cases: Tuple[str, ...] = None
    holdout_cases: Tuple[str, ...] = None

    epochs: int = 300
    batch_size: int = 64           # 一个 batch 含多少个「窗口样本」
    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_decay_step: int = 50
    lr_decay_gamma: float = 0.9

    use_amp: bool = True           # 混合精度（4060 支持，提速省显存）
    num_workers: int = 0           # 数据已在内存/显存，无需多进程
    seed: int = 42

    print_every: int = 10
    save_every: int = 50

    lambda_nonneg: float = 0.05    # 非负约束权重（PSD 物理上 >= 0）
    lambda_conc: float = 1.0       # 浓度预测损失权重


@dataclass
class Config:
    """顶层配置：聚合四组子配置。"""

    path: PathConfig = field(default_factory=PathConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def branch_dim(self) -> int:
        """Branch = T_hist(n_in) + C_hist(n_in) + PSD_hist(n_in×n_L) + T_plan(n_out)。"""
        w = self.window
        n_in, n_out = w.n_in, w.n_out
        psd_hist = n_in * w.n_L_sensors
        return n_in + n_in + psd_hist + n_out

    @property
    def n_query(self) -> int:
        """Trunk 查询点数 = 粒径评估点 × 输出时间点。"""
        return self.window.n_L_eval * self.window.n_out


def get_default_config() -> Config:
    """返回默认配置实例。"""
    return Config()


def apply_run_config(cfg: Config, run_dir: str) -> Config:
    """用训练 run 目录下的 ``config.json`` 覆盖 window/model（评估/推理用）。"""
    import json

    path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(path):
        return cfg
    with open(path, "r", encoding="utf-8") as fp:
        j = json.load(fp)
    w = j.get("window", {})
    m = j.get("model", {})
    for key, val in w.items():
        if hasattr(cfg.window, key):
            setattr(cfg.window, key, val)
    for key, val in m.items():
        if hasattr(cfg.model, key):
            setattr(cfg.model, key, val)
    return cfg
