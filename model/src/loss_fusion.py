import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def evidential_loss(u, la, alpha, beta, y, weight_reg=1.0):
    om = 2 * beta * (1 + la)
    err = u - y
    maha = torch.sum(la * err ** 2, dim=1, keepdim=True)

    nll = (
        0.5 * torch.log(torch.tensor(np.pi, device=la.device) / la)
        - alpha * torch.log(om)
        + (alpha + 0.5) * torch.log(maha + om)
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    )

    reg = weight_reg * torch.mean(torch.abs(u - y) * (2 * la + alpha))  # 显式正则项
    return nll.mean() + reg


class FusionCloudLoss(nn.Module):
    """
    云去除任务三路融合 loss：Evidential Loss + L1 Loss
    total = evidential + l1，各自按融合权重加和
    """
    def __init__(self, config):
        super().__init__()
        self.lambda1 = getattr(config, 'lambda1', 1.0)
        self.lambda2 = getattr(config, 'lambda2', 1.0)
        self.weight_reg = getattr(config, 'weight_reg', 0.0)
        self.l1_criterion = nn.SmoothL1Loss()

    def forward(self, out_un, out_mamba, out_fused, y_gt):
        # --- evidential losses ---
        loss_un_evi = evidential_loss(
            out_un['delta'], out_un['gamma'], out_un['alpha'], out_un['beta'],
            y_gt, weight_reg=self.weight_reg
        )
        loss_mamba_evi = evidential_loss(
            out_mamba['delta'], out_mamba['gamma'], out_mamba['alpha'], out_mamba['beta'],
            y_gt, weight_reg=self.weight_reg
        )
        loss_fused_evi = evidential_loss(
            out_fused['delta'], out_fused['gamma'], out_fused['alpha'], out_fused['beta'],
            y_gt, weight_reg=self.weight_reg
        )

        # --- l1 losses ---
        loss_un_l1 = self.l1_criterion(out_un['delta'], y_gt)
        loss_mamba_l1 = self.l1_criterion(out_mamba['delta'], y_gt)
        loss_fused_l1 = self.l1_criterion(out_fused['delta'], y_gt)

        # --- total ---
        total_loss = (
            loss_fused_evi + self.lambda1 * loss_mamba_evi + self.lambda2 * loss_un_evi +
            loss_fused_l1 + self.lambda1 * loss_mamba_l1 + self.lambda2 * loss_un_l1
        )
        loss_un = loss_un_l1+loss_un_evi
        loss_ma = loss_mamba_l1+loss_mamba_evi
        loss_fused = loss_fused_l1+loss_fused_evi
        aux = {
            'loss_un': loss_un.item(),
            'loss_mamba': loss_ma.item(),
            'loss_fused': loss_fused.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, aux
