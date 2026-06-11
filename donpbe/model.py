"""
DeepONet 模型模块（PyTorch 实现，多任务：PSD + 浓度）。

结构：
    Branch 网络：输入历史/条件向量 (B, branch_dim) → 潜向量 (B, p)
    Trunk_PSD：  输入查询坐标 (Q_psd, 2)=[L, τ]  → (Q_psd, p)
    Trunk_Conc： 输入查询坐标 (Q_c, 1)=[τ]       → (Q_c, p)
    PSD 输出：   branch @ trunk_psd.T + bias_psd  → (B, Q_psd) ≈ n(L, τ)
    浓度输出：   branch @ trunk_conc.T + bias_conc → (B, Q_c)   ≈ C(τ)

PSD 与浓度均为 DeepONet 算子形式（Branch·Trunk 内积），
不在 Branch 上接 MLP 直接回归浓度。
"""

from typing import List

import torch
import torch.nn as nn


def _make_mlp(in_dim: int, hiddens: List[int], out_dim: int,
              activation: str = "tanh", last_activation: bool = False) -> nn.Sequential:
    """构建多层感知机。"""
    act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU,
               "silu": nn.SiLU}
    Act = act_map.get(activation, nn.Tanh)
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hiddens:
        layers.append(nn.Linear(prev, h))
        layers.append(Act())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if last_activation:
        layers.append(Act())
    return nn.Sequential(*layers)


class BranchNet(nn.Module):
    """Branch 网络：编码历史 T/C、PSD 历史 n(L,t) 与输出窗计划温度。"""

    def __init__(self, in_dim: int, hiddens: List[int], latent_dim: int,
                 activation: str = "tanh"):
        super().__init__()
        self.net = _make_mlp(in_dim, hiddens, latent_dim, activation,
                             last_activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrunkNet(nn.Module):
    """Trunk 网络：编码查询坐标 (L, τ)，末层加激活作为基函数。"""

    def __init__(self, in_dim: int, hiddens: List[int], latent_dim: int,
                 activation: str = "tanh"):
        super().__init__()
        self.net = _make_mlp(in_dim, hiddens, latent_dim, activation,
                             last_activation=True)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


class DeepONet(nn.Module):
    """Deep Operator Network。

    Parameters
    ----------
    branch_dim : int
        Branch 输入维度。
    trunk_dim : int
        Trunk 输入维度（本项目为 2：L 与 τ）。
    branch_hiddens, trunk_hiddens : list[int]
        两个子网各隐藏层宽度。
    latent_dim : int
        公共潜在维度 p。
    """

    def __init__(self, branch_dim: int, trunk_dim: int = 2,
                 branch_hiddens: List[int] = None,
                 trunk_hiddens: List[int] = None,
                 conc_trunk_hiddens: List[int] = None,
                 latent_dim: int = 128, activation: str = "tanh",
                 n_out: int = 100):
        super().__init__()
        branch_hiddens = branch_hiddens or [256, 256, 256]
        trunk_hiddens = trunk_hiddens or [128, 128, 128]
        conc_trunk_hiddens = conc_trunk_hiddens or trunk_hiddens
        self.n_out = n_out
        self.branch = BranchNet(branch_dim, branch_hiddens, latent_dim, activation)
        self.trunk = TrunkNet(trunk_dim, trunk_hiddens, latent_dim, activation)
        # 浓度算子：Trunk 仅编码输出窗口内时间坐标 τ
        self.trunk_conc = TrunkNet(1, conc_trunk_hiddens, latent_dim, activation)
        self.bias = nn.Parameter(torch.zeros(1))
        self.bias_conc = nn.Parameter(torch.zeros(1))

    def forward(self, branch_in: torch.Tensor,
                trunk_in: torch.Tensor, trunk_conc_in: torch.Tensor):
        """
        Parameters
        ----------
        branch_in      : (B, branch_dim)
        trunk_in       : (Q_psd, 2)  固定 PSD 查询网格 [L, τ]
        trunk_conc_in  : (Q_c, 1)    固定浓度查询网格 [τ]

        Returns
        -------
        psd  : (B, Q_psd)  归一化 PSD n(L, τ)
        conc : (B, Q_c)    归一化浓度 C(τ)，Q_c = n_out
        """
        b = self.branch(branch_in)              # (B, p)
        t = self.trunk(trunk_in)                # (Q_psd, p)
        tc = self.trunk_conc(trunk_conc_in)     # (Q_c, p)
        psd = b @ t.t() + self.bias             # (B, Q_psd)
        conc = b @ tc.t() + self.bias_conc      # (B, Q_c)
        return psd, conc

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
