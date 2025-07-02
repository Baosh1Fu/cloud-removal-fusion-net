import torch
import torch.nn as nn
from torch import Tensor
from torch.overrides import has_torch_function_variadic, handle_torch_function
from functorch import vmap  
import numpy as np

def evidential_loss(u, la, alpha, beta, y, weight_reg=1.0):
    om = 2 * beta * (1 + la)

    nll = (
        0.5 * torch.log(np.pi / la)
        - alpha * torch.log(om)
        + (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om)
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    )

    loss = torch.mean(nll)
    lossr = weight_reg * torch.mean(torch.abs(u - y) * (2 * la + alpha))

    return loss + lossr

class FusionCloudLoss(nn.Module):
    """
    云去除任务三路融合 NIG Loss（含 mask 和正则项）
    """
    def __init__(self, config):
        super().__init__()
        self.lambda1 = getattr(config, 'lambda1', 1.0)
        self.lambda2 = getattr(config, 'lambda2', 1.0)
        self.weight_reg = getattr(config, 'weight_reg', 0.0)

    def forward(self, out_un, out_mamba, out_fused, y_gt):
        loss_un = evidential_loss(
            out_un['delta'], out_un['gamma'], out_un['alpha'], out_un['beta'],
            y_gt,  weight_reg=self.weight_reg
        )

        loss_mamba = evidential_loss(
            out_mamba['delta'], out_mamba['gamma'], out_mamba['alpha'], out_mamba['beta'],
            y_gt,  weight_reg=self.weight_reg
        )

        loss_fused = evidential_loss(
            out_fused['delta'], out_fused['gamma'], out_fused['alpha'], out_fused['beta'],
            y_gt,  weight_reg=self.weight_reg
        )

        total_loss = loss_fused + self.lambda1 * loss_mamba + self.lambda2 * loss_un

        aux = {
            'loss_un': loss_un.item(),
            'loss_mamba': loss_mamba.item(),
            'loss_fused': loss_fused.item()
        }

        return total_loss, aux

