import torch
import torch.nn as nn
from torch import Tensor
from torch.overrides import has_torch_function_variadic, handle_torch_function
from functorch import vmap  

def multi_diag_gaussian_nll(pred, target, var):
    pred, target, var = pred.squeeze(dim=1), target.squeeze(dim=1), var.squeeze(dim=1)
    k = pred.shape[-1]
    prec = torch.diag_embed(1 / var, offset=0, dim1=-2, dim2=-1)
    logdetv = var.log().sum()
    err = (pred - target).unsqueeze(dim=1)
    xTCx = torch.bmm(torch.bmm(err, prec), err.permute(0, 2, 1)).squeeze().nan_to_num().clamp(min=1e-9)
    loss = -(-k / 2 * torch.log(2 * torch.tensor(torch.pi)) - 1 / 2 * logdetv - 1 / 2 * xTCx)
    return loss, torch.diag_embed(var, offset=0, dim1=-2, dim2=-1).cpu()

def multi_gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
    chunk=None
) -> Tensor:
    if has_torch_function_variadic(input, target, var):
        return handle_torch_function(
            multi_gaussian_nll_loss,
            (input, target, var),
            input, target, var,
            full=full, eps=eps, reduction=reduction, chunk=chunk
        )

    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    mapdims = (-1, -1, -1)
    loss, variance = vmap(vmap(multi_diag_gaussian_nll, in_dims=mapdims, chunk_size=chunk), in_dims=mapdims, chunk_size=chunk)(input, target, var)

    variance = variance.moveaxis(1, -1).moveaxis(0, -1).unsqueeze(1)

    if reduction == 'mean':
        return loss.mean(), variance
    elif reduction == 'sum':
        return loss.sum(), variance
    else:
        return loss, variance

class FusionCloudLoss(nn.Module):
    """
    云去除任务的三路融合 NLL 损失：
    总损失 = L_un + λ1 * L_mamba + λ2 * L_fused。
    返回 (total_loss, aux_dict)，aux_dict 包含每路协方差张量和各路标量损失。
    """
    def __init__(self, config):
        super().__init__()
        self.lambda1 = getattr(config, 'lambda1', 1.0)
        self.lambda2 = getattr(config, 'lambda2', 1.0)
        self.full = True
        self.eps = 1e-8
        self.reduction = 'mean'
        self.chunk = None

    def forward(self, out_un, out_mamba, out_fused, y_gt):
        var_un = out_un['beta'] / (out_un['alpha'] - 1.0).clamp(min=self.eps)
        var_mamba = out_mamba['beta'] / (out_mamba['alpha'] - 1.0).clamp(min=self.eps)
        var_fused = out_fused['beta'] / (out_fused['alpha'] - 1.0).clamp(min=self.eps)

        loss_un, varmat_un = multi_gaussian_nll_loss(out_un['delta'], y_gt, var_un,
                                                     full=self.full, eps=self.eps, reduction=self.reduction, chunk=self.chunk)
        loss_mamba, varmat_mamba = multi_gaussian_nll_loss(out_mamba['delta'], y_gt, var_mamba,
                                                           full=self.full, eps=self.eps, reduction=self.reduction, chunk=self.chunk)
        loss_fused, varmat_fused = multi_gaussian_nll_loss(out_fused['delta'], y_gt, var_fused,
                                                           full=self.full, eps=self.eps, reduction=self.reduction, chunk=self.chunk)

        total_loss = loss_fused + self.lambda1 * loss_mamba + self.lambda2 * loss_un

        aux = {
            'var_un':     varmat_un,
            'var_mamba':  varmat_mamba,
            'var_fused':  varmat_fused,
            'loss_un':    loss_un.item(),
            'loss_mamba': loss_mamba.item(),
            'loss_fused': loss_fused.item(),
        }

        return total_loss, aux

