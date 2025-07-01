import torch
import torch.nn as nn
from src.backbones import uncrtaints_fusion, mamba_fusion

class FusionUNCRTAINTS(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 波段数量
        S1_BANDS = 2
        S2_BANDS = 13
        in_channels = S1_BANDS * config.use_sar + S2_BANDS

        # — UNCRTAINTS 分支参数 — #
        un_args = {
            'input_dim':    in_channels,
            'encoder_widths': config.encoder_widths,
            'decoder_widths': config.decoder_widths,
            'agg_mode':       config.agg_mode,
            'encoder_norm':   config.encoder_norm,
            'decoder_norm':   config.decoder_norm,
            'n_head':         config.n_head,
            'd_model':        config.d_model,
            'd_k':            config.d_k,
            'pad_value':      config.pad_value,
            'positional_encoding': config.positional_encoding,
            'use_v':          config.use_v,
            'block_type':     config.block_type,
            'is_mono':        config.pretrain,
        }
        self.un_model = uncrtaints_fusion.UNCRTAINTS(**un_args)

        # Mamba3D 分支参数 #
        mamba_args = {
            'input_dim':     in_channels,
            'output_dim':    S2_BANDS,
            'encoder_widths': config.encoder_widths,
            'decoder_widths': config.decoder_widths,
            'd_state':       getattr(config, 'd_state', 16),
            'covmode':       config.covmode,
            'scale_by':      config.scale_by,
            'norm_type':     config.encoder_norm,
        }
        self.mamba_model = mamba_fusion.UNCRTAINTS_Mamba3D(**mamba_args)

    def moe_nig(self, u1, la1, a1, b1, u2, la2, a2, b2):
        la = la1 + la2
        u = (la1 * u1 + la2 * u2) / (la + 1e-8)
        a = a1 + a2 + 0.5
        b = b1 + b2 + 0.5 * (la1 * (u1 - u) ** 2 + la2 * (u2 - u) ** 2)
        return u, la, a, b

    def compute_uncertainty(self, u, la, a, b):
        aleatoric = b / (a - 1 + 1e-8)
        epistemic = aleatoric / (la + 1e-8)
        total = aleatoric + epistemic
        return aleatoric, epistemic, total

    def forward(self, x, batch_positions=None):
        # — 两个分支各自前向传播 — #
        out_un = self.un_model(x, batch_positions)
        out_ma = self.mamba_model(x)

        # — 读取 NIG 参数 — #
        u1, la1, a1, b1 = out_un['delta'], out_un['gamma'], out_un['alpha'], out_un['beta']
        u2, la2, a2, b2 = out_ma['delta'], out_ma['gamma'], out_ma['alpha'], out_ma['beta']

        # — MoE 融合 — #
        u_f, la_f, a_f, b_f = self.moe_nig(u1, la1, a1, b1, u2, la2, a2, b2)

        # — 不确定度分解 — #
        ale_f, epi_f, tot_f = self.compute_uncertainty(u_f, la_f, a_f, b_f)

        # — 输出结构 — #
        return {
            # 原始分支 UN
            'delta_un': u1, 'gamma_un': la1, 'alpha_un': a1, 'beta_un': b1,
            # 原始分支 Mamba
            'delta_ma': u2, 'gamma_ma': la2, 'alpha_ma': a2, 'beta_ma': b2,
            # 融合结果
            'delta_fuse': u_f, 'gamma_fuse': la_f, 'alpha_fuse': a_f, 'beta_fuse': b_f,
            # 不确定性分解
            'aleatoric_fuse': ale_f,
            'epistemic_fuse': epi_f,
            'total_uncertainty_fuse': tot_f
        }
