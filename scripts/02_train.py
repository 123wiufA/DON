"""
脚本 02：训练 DeepONet（前 15min → 后 5min PSD 算子）。

用法::
    python scripts/02_train.py
    python scripts/02_train.py --epochs 500 --batch 128 --lr 5e-4
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from donpbe.config import get_default_config
from donpbe.device import setup_device, set_seed
from donpbe.dataset import PBEWindowData
from donpbe.model import DeepONet
from donpbe.trainer import Trainer
from donpbe.utils import plot_loss_curve


def main():
    cfg = get_default_config()
    ap = argparse.ArgumentParser(description="训练 DeepONet")
    ap.add_argument("--npz", default=cfg.path.npz_path)
    ap.add_argument("--epochs", type=int, default=cfg.train.epochs)
    ap.add_argument("--batch", type=int, default=cfg.train.batch_size)
    ap.add_argument("--lr", type=float, default=cfg.train.lr)
    args = ap.parse_args()

    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch
    cfg.train.lr = args.lr

    set_seed(cfg.train.seed)
    device = setup_device()

    if not os.path.isfile(args.npz):
        raise FileNotFoundError(
            f"找不到数据集 {args.npz}\n请先运行: python scripts/01_preprocess.py")

    # ---- 数据 ----
    data = PBEWindowData(args.npz, cfg)
    data.summary()
    train_data = data.build_split("train")
    val_data = data.build_split("val")
    print(f"[Data] 训练样本={train_data[0].shape[0]}, 测试样本={val_data[0].shape[0]}  "
          f"(PSD标签 {train_data[1].shape}, 浓度标签 {train_data[2].shape})")

    # ---- 输出目录 ----
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg.path.results_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Run] {run_tag} -> {out_dir}")

    data.save_norm_params(os.path.join(out_dir, "norm_params.npz"))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "branch_dim": cfg.branch_dim, "n_query": cfg.n_query,
            "window": cfg.window.__dict__, "model": cfg.model.__dict__,
            "train": {k: list(v) if isinstance(v, tuple) else v
                      for k, v in cfg.train.__dict__.items()},
        }, f, indent=2, ensure_ascii=False)

    # ---- 模型 ----
    model = DeepONet(
        branch_dim=cfg.branch_dim, trunk_dim=2,
        branch_hiddens=cfg.model.branch_hiddens,
        trunk_hiddens=cfg.model.trunk_hiddens,
        latent_dim=cfg.model.latent_dim,
        activation=cfg.model.activation,
        n_out=cfg.window.n_out,
        conc_trunk_hiddens=cfg.model.conc_trunk_hiddens)

    # ---- 训练 ----
    trainer = Trainer(model, data.trunk_grid, data.trunk_conc_grid, device, cfg)
    trainer.fit(train_data, val_data, save_dir=os.path.join(out_dir, "weights"))

    # ---- 损失曲线 ----
    plot_loss_curve(trainer.train_hist, trainer.val_hist,
                    os.path.join(out_dir, "loss_curve.png"))
    print(f"[OK] 训练结束，结果在 {out_dir}")


if __name__ == "__main__":
    main()
