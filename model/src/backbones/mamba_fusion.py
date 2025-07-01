import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba

S2_BANDS = 13

# ---------------------------- Utility Functions ----------------------------
def get_norm_layer(channels, num_feats, n_groups=4, layer_type='group'):
    if layer_type == 'batch':
        return nn.BatchNorm2d(channels)
    elif layer_type == 'instance':
        return nn.InstanceNorm2d(channels)
    elif layer_type == 'group':
        return nn.GroupNorm(num_channels=num_feats, num_groups=n_groups)
    else:
        return nn.Identity()

def monig_fusion(pred_list, weights=None):
    n = len(pred_list)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        weights = [w / sum(weights) for w in weights]

    deltas = torch.stack([p['delta'] for p in pred_list], dim=0)
    gammas = torch.stack([p['gamma'] for p in pred_list], dim=0)
    alphas = torch.stack([p['alpha'] for p in pred_list], dim=0)
    betas  = torch.stack([p['beta']  for p in pred_list], dim=0)

    w = torch.tensor(weights, device=deltas.device).view(n, 1, 1, 1, 1, 1)

    fused_delta = torch.sum(w * deltas, dim=0)
    fused_gamma = torch.sum(w * gammas, dim=0)
    fused_alpha = torch.sum(w * alphas, dim=0)
    fused_beta  = torch.sum(w * betas,  dim=0)

    return {
        'delta': fused_delta,
        'gamma': fused_gamma,
        'alpha': fused_alpha,
        'beta':  fused_beta
    }

# ---------------------------- Blocks ----------------------------
class ResidualConv(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=chan),
            nn.GELU(),
            nn.Conv2d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=chan),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2.0):
        super().__init__()
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.forward_mamba_t = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.backward_mamba_t = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.forward_mamba_t2 = Mamba(d_model=d_model, d_state=d_state, expand=expand)
        self.backward_mamba_t2 = Mamba(d_model=d_model, d_state=d_state, expand=expand)

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.local_res1 = ResidualConv(d_model)
        self.depthwise = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pointwise = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.local_res2 = ResidualConv(d_model)

    def forward(self, x):
        B, D, H, W = x.shape
        x_flat = rearrange(x, 'b d h w -> b (h w) d')
        fwd = self.forward_mamba(x_flat)
        bwd = torch.flip(self.backward_mamba(torch.flip(x_flat, dims=[1])), dims=[1])
        x_spatial = 0.5 * (fwd + bwd)
        x_spatial = rearrange(x_spatial, 'b (h w) d -> b d h w', h=H, w=W)

        x_local = self.local_res1(x_spatial)
        x_local = self.depthwise(x_local)
        x_local = self.act(x_local)
        x_local = self.pointwise(x_local)
        x_local = self.local_res2(x_local)

        x_time = rearrange(x, 'b d h w -> b d (h w)')
        x_time = rearrange(x_time, 'b d s -> b s d')
        t_fwd = self.forward_mamba_t(x_time)
        t_bwd = torch.flip(self.backward_mamba_t(torch.flip(x_time, dims=[1])), dims=[1])
        x_time_global = 0.5 * (t_fwd + t_bwd)
        x_time_global = rearrange(x_time_global, 'b (h w) d -> b d h w', h=H, w=W)

        x_ds = self.downsample(x)
        H2, W2 = H // 2, W // 2
        ds_flat = rearrange(x_ds, 'b d h w -> b d (h w)')
        ds_flat = rearrange(ds_flat, 'b d s -> b s d')
        ds_fwd = self.forward_mamba_t2(ds_flat)
        ds_bwd = torch.flip(self.backward_mamba_t2(torch.flip(ds_flat, dims=[1])), dims=[1])
        x_time_ds = 0.5 * (ds_fwd + ds_bwd)
        x_time_ds = rearrange(x_time_ds, 'b (h w) d -> b d h w', h=H2, w=W2)
        x_time_ds = self.upsample(x_time_ds)

        x_time = 0.5 * x_time_global + 0.5 * x_time_ds
        return x_spatial + x_local + x_time

class TemporalEncoder(nn.Module):
    def __init__(self, widths, d_state=16, expand=2.0, norm_type='group', n_groups=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.mamba_blocks = nn.ModuleList()
        for i, out_c in enumerate(widths):
            in_c = widths[i - 1] if i > 0 else widths[0]
            self.layers.append(nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                get_norm_layer(out_c, out_c, n_groups, norm_type),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                get_norm_layer(out_c, out_c, n_groups, norm_type),
                nn.ReLU(inplace=True),
            ))
            self.mamba_blocks.append(MambaBlock(out_c, d_state=d_state, expand=expand))
        self.final_encoder_conv = nn.Sequential(
            nn.Conv3d(widths[-1], widths[-1], kernel_size=3, padding=1),
            get_norm_layer(widths[-1], widths[-1], n_groups, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv3d(widths[-1], widths[-1], kernel_size=3, padding=1),
            get_norm_layer(widths[-1], widths[-1], n_groups, norm_type),
            nn.ReLU(inplace=True),
        )
        self.final_mamba_block = MambaBlock(widths[-1], d_state=d_state, expand=expand)

    def forward(self, x):
        skips = []
        for conv, mamba in zip(self.layers, self.mamba_blocks):
            x = conv(x)
            x_tavg = x.mean(dim=2)
            x = mamba(x_tavg) + x_tavg
            skips.append(x)
        x = x.unsqueeze(2)
        x = self.final_encoder_conv(x)
        x_tavg = x.mean(dim=2)
        x = self.final_mamba_block(x_tavg) + x_tavg
        skips.append(x)
        return x, skips

class UNCRTAINTS_Mamba3D(nn.Module):
    def __init__(self, input_dim, output_dim=S2_BANDS, encoder_widths=[128, 128], decoder_widths=[128, 128], d_state=16, covmode='diag', scale_by=1.0, norm_type='group'):
        super().__init__()
        self.output_dim = output_dim
        self.scale_by = scale_by
        self.covmode = covmode
        self.input_projection = nn.Conv3d(input_dim, encoder_widths[0], kernel_size=3, padding=1)
        self.pre_encoder_conv = nn.Sequential(
            nn.Conv3d(encoder_widths[0], encoder_widths[0], kernel_size=3, padding=1),
            get_norm_layer(encoder_widths[0], encoder_widths[0], n_groups=4, layer_type=norm_type),
            nn.ReLU(inplace=True),
            nn.Conv3d(encoder_widths[0], encoder_widths[0], kernel_size=3, padding=1),
            get_norm_layer(encoder_widths[0], encoder_widths[0], n_groups=4, layer_type=norm_type),
            nn.ReLU(inplace=True),
        )
        self.encoder = TemporalEncoder(widths=encoder_widths, d_state=d_state, expand=2.0, norm_type=norm_type)
        self.decoder = Decoder(decoder_widths, output_dim=output_dim, covmode=covmode, norm_type=norm_type)

    def forward(self, x, batch_positions=None):
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.input_projection(x)
        x = self.pre_encoder_conv(x)
        feat, skips = self.encoder(x)
        delta, gamma, alpha, beta, aux = self.decoder(feat, skips)

        delta = delta.view(B, 1, self.output_dim, H, W)
        gamma = gamma.view(B, 1, self.output_dim, H, W)
        alpha = alpha.view(B, 1, self.output_dim, H, W)
        beta  = beta.view(B, 1, self.output_dim, H, W)

        aux_outputs = []
        for d in aux:
            aux_outputs.append({
                'delta': d['delta'].view(B, 1, self.output_dim, H, W),
                'gamma': d['gamma'].view(B, 1, self.output_dim, H, W),
                'alpha': d['alpha'].view(B, 1, self.output_dim, H, W),
                'beta':  d['beta'].view(B, 1, self.output_dim, H, W),
            })

        output = monig_fusion(aux_outputs + [{
            'delta': delta,
            'gamma': gamma,
            'alpha': alpha,
            'beta': beta
        }])

        return output
