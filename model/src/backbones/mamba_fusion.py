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

# ---------------------------- Decoder ----------------------------
class Decoder(nn.Module):
    def __init__(self, decoder_widths, out_channels, covmode='diag', norm_type='group', n_groups=4):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_widths)):
            in_c = decoder_widths[i - 1] if i > 0 else decoder_widths[0]
            out_c = decoder_widths[i]
            self.decoder_blocks.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                get_norm_layer(out_c, out_c, n_groups, norm_type),
                nn.ReLU(inplace=True)
            ))

        self.delta_head = nn.Conv2d(decoder_widths[-1], out_channels, 1)
        self.gamma_head = nn.Sequential(
            nn.Conv2d(decoder_widths[-1], out_channels, 1),
            nn.Softplus()
        )
        self.alpha_head = nn.Sequential(
            nn.Conv2d(decoder_widths[-1], out_channels, 1),
            nn.Softplus()
        )

        if covmode == 'diag':
            beta_out_dim = out_channels
        elif covmode == 'iso':
            beta_out_dim = 1
        elif covmode == 'uni':
            beta_out_dim = out_channels
        elif covmode == 'full':
            beta_out_dim = out_channels * out_channels
        else:
            raise ValueError(f"Unsupported covmode: {covmode}")

        self.beta_head = nn.Sequential(
            nn.Conv2d(decoder_widths[-1], beta_out_dim, 1),
            nn.Softplus()
        )

    def forward(self, x, skips=None):
        for block in self.decoder_blocks:
            x = block(x)

        delta = self.delta_head(x)
        gamma = self.gamma_head(x)
        alpha = self.alpha_head(x)
        beta  = self.beta_head(x)

        return delta, gamma, alpha, beta

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

# ---------------------------- Temporal Encoder ----------------------------
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
        for conv, mamba in zip(self.layers, self.mamba_blocks):
            x = conv(x)
            x_tavg = x.mean(dim=2)
            x = mamba(x_tavg) + x_tavg
        x = x.unsqueeze(2)
        x = self.final_encoder_conv(x)
        x_tavg = x.mean(dim=2)
        x = self.final_mamba_block(x_tavg) + x_tavg
        return x

# ---------------------------- Full Model ----------------------------
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
        self.decoder = Decoder(decoder_widths, out_channels=output_dim, covmode=covmode, norm_type=norm_type)

    def forward(self, x, batch_positions=None):
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.input_projection(x)
        x = self.pre_encoder_conv(x)
        feat = self.encoder(x)
        delta, gamma, alpha, beta = self.decoder(feat, skips=None)

        return {
            'delta': delta.view(B, 1, self.output_dim, H, W),
            'gamma': gamma.view(B, 1, self.output_dim, H, W),
            'alpha': alpha.view(B, 1, self.output_dim, H, W),
            'beta':  beta.view(B, 1, self.output_dim, H, W),
        }
