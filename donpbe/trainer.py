"""
训练器模块。

负责 DeepONet 多任务的训练循环：
  - 数据常驻显存（样本量不大），用索引切 batch，省去 DataLoader 开销
  - 联合损失 = PSD_MSE + λ_nonneg·非负惩罚(PSD≥0) + λ_conc·浓度_MSE
  - 混合精度 (AMP)，适配 RTX 4060
  - StepLR 学习率衰减
  - 记录 train/val 历史，保存最优与定期检查点
"""

import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class Trainer:
    """DeepONet 训练器。"""

    def __init__(self, model: nn.Module, trunk_grid: np.ndarray,
                 trunk_conc_grid: np.ndarray,
                 device: torch.device, cfg):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.tcfg = cfg.train

        # 固定 Trunk 网格常驻显存（PSD 与浓度各一套）
        self.trunk = torch.from_numpy(trunk_grid).to(device)
        self.trunk_conc = torch.from_numpy(trunk_conc_grid).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.tcfg.lr,
            weight_decay=self.tcfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.tcfg.lr_decay_step,
            gamma=self.tcfg.lr_decay_gamma)
        self.use_amp = bool(self.tcfg.use_amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.train_hist = []
        self.val_hist = []
        self.best_val = float("inf")
        self.best_epoch = -1

    def _to_device(self, branch, psd, conc):
        return (torch.from_numpy(branch).to(self.device),
                torch.from_numpy(psd).to(self.device),
                torch.from_numpy(conc).to(self.device))

    def _loss(self, pred_psd, tgt_psd, pred_conc, tgt_conc):
        mse_psd = torch.mean((pred_psd - tgt_psd) ** 2)
        nonneg = torch.mean(torch.relu(-pred_psd) ** 2)
        mse_conc = torch.mean((pred_conc - tgt_conc) ** 2)
        total = (mse_psd + self.tcfg.lambda_nonneg * nonneg
                 + self.tcfg.lambda_conc * mse_conc)
        return total, mse_psd, mse_conc

    @torch.no_grad()
    def evaluate(self, branch, psd, conc) -> Tuple[float, float]:
        """整批评估 PSD 与浓度的 MSE（归一化尺度）。"""
        if branch.shape[0] == 0:
            return float("nan"), float("nan")
        self.model.eval()
        p_list, c_list = [], []
        bs = 256
        for i in range(0, branch.shape[0], bs):
            with torch.autocast("cuda", enabled=self.use_amp):
                p, c = self.model(branch[i:i + bs], self.trunk, self.trunk_conc)
            p_list.append(p.float())
            c_list.append(c.float())
        pred_psd = torch.cat(p_list, 0)
        pred_conc = torch.cat(c_list, 0)
        mse_psd = float(torch.mean((pred_psd - psd) ** 2).item())
        mse_conc = float(torch.mean((pred_conc - conc) ** 2).item())
        return mse_psd, mse_conc

    def fit(self, train_data, val_data, save_dir: str) -> None:
        b_tr, p_tr, c_tr = self._to_device(*train_data)
        has_val = val_data is not None and val_data[0].shape[0] > 0
        if has_val:
            b_va, p_va, c_va = self._to_device(*val_data)

        n = b_tr.shape[0]
        bs = self.tcfg.batch_size
        os.makedirs(save_dir, exist_ok=True)

        print(f"[Train] 样本数={n}, batch={bs}, epochs={self.tcfg.epochs}, "
              f"AMP={self.use_amp}, 参数量={self.model.count_params():,}")

        for epoch in range(1, self.tcfg.epochs + 1):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            epoch_loss, n_batch = 0.0, 0
            t0 = time.time()

            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                bb, pp, cc = b_tr[idx], p_tr[idx], c_tr[idx]
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=self.use_amp):
                    pred_psd, pred_conc = self.model(bb, self.trunk, self.trunk_conc)
                    loss, _, _ = self._loss(pred_psd, pp, pred_conc, cc)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                epoch_loss += loss.item()
                n_batch += 1

            self.scheduler.step()
            train_loss = epoch_loss / max(n_batch, 1)
            self.train_hist.append(train_loss)

            if has_val:
                v_psd, v_conc = self.evaluate(b_va, p_va, c_va)
                val_loss = v_psd + self.tcfg.lambda_conc * v_conc
            else:
                val_loss = train_loss
            self.val_hist.append(val_loss)

            if val_loss < self.best_val:
                self.best_val = val_loss
                self.best_epoch = epoch
                torch.save(self.model.state_dict(),
                           os.path.join(save_dir, "best.pt"))

            if epoch % self.tcfg.print_every == 0 or epoch == 1:
                dt = time.time() - t0
                lr = self.optimizer.param_groups[0]["lr"]
                tag = "val" if has_val else "train(no val)"
                print(f"  epoch {epoch:4d}/{self.tcfg.epochs}  "
                      f"train={train_loss:.4e}  {tag}={val_loss:.4e}  "
                      f"lr={lr:.2e}  {dt:.1f}s")

            if epoch % self.tcfg.save_every == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(save_dir, f"ckpt_epoch_{epoch:04d}.pt"))

        torch.save(self.model.state_dict(), os.path.join(save_dir, "final.pt"))
        print(f"[Train] 完成。最优 epoch={self.best_epoch}, "
              f"best_val={self.best_val:.4e}")
