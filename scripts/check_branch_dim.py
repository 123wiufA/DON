"""快速校验 Branch 维数为 39100。"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from donpbe.config import get_default_config
from donpbe.dataset import PBEWindowData

cfg = get_default_config()
assert cfg.branch_dim == 39100, cfg.branch_dim
data = PBEWindowData(cfg.path.npz_path, cfg)
assert len(data.T_sensor_idx) == 300
assert len(data.T_future_sensor_idx) == 100
b = data._build_branch(0, 0)
assert b.shape == (39100,), b.shape
print(f"branch_dim={cfg.branch_dim}  sample_branch={b.shape}  OK")
