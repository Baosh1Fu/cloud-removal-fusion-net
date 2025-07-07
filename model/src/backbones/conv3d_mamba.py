import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba


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
        x_fwd = self.forward_mamba(x_flat)
        x_bwd = torch.flip(self.backward_mamba(torch.flip(x_flat, dims=[1])), dims=[1])
        x_spatial = 0.5 * (x_fwd + x_bwd)
        x_spatial = rearrange(x_spatial, 'b (h w) d -> b d h w', h=H, w=W)

        x_local = self.local_res1(x_spatial)
        x_local = self.depthwise(x_local)
        x_local = self.act(x_local)
        x_local = self.pointwise(x_local)
        x_local = self.local_res2(x_local)

        x_time = rearrange(x, 'b d h w -> b d (h w)')
        x_time = rearrange(x_time, 'b d s -> b s d')
        x_time_fwd = self.forward_mamba_t(x_time)
        x_time_bwd = torch.flip(self.backward_mamba_t(torch.flip(x_time, dims=[1])), dims=[1])
        x_time_global = 0.5 * (x_time_fwd + x_time_bwd)
        x_time_global = rearrange(x_time_global, 'b (h w) d -> b d h w', h=H, w=W)

        x_ds = self.downsample(x)
        H2, W2 = H // 2, W // 2
        x_ds_flat = rearrange(x_ds, 'b d h w -> b (h w) d')
        x_ds_fwd = self.forward_mamba_t2(x_ds_flat)
        x_ds_bwd = torch.flip(self.backward_mamba_t2(torch.flip(x_ds_flat, dims=[1])), dims=[1])
        x_time_ds = 0.5 * (x_ds_fwd + x_ds_bwd)
        x_time_ds = rearrange(x_time_ds, 'b (h w) d -> b d h w', h=H2, w=W2)
        x_time_ds = self.upsample(x_time_ds)

        x_time = 0.5 * x_time_global + 0.5 * x_time_ds

        return x_spatial + x_local + x_time


class TemporalEncoder(nn.Module):
    def __init__(self, widths, d_state=16, expand=2.0, norm_type='group', n_groups=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.mamba_blocks = nn.ModuleList()

        for i in range(len(widths)):
            in_c = widths[i - 1] if i > 0 else widths[0]
            out_c = widths[i]
            self.layers.append(nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                get_norm_layer(out_c, out_c, n_groups, norm_type),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                get_norm_layer(out_c, out_c, n_groups, norm_type),
                nn.ReLU(inplace=True),
            ))
            self.mamba_blocks.append(MambaBlock(out_c, d_state=d_state, expand=expand))

        self.final_encoder_conv3d = nn.Sequential(
            nn.Conv3d(widths[-1], widths[-1], kernel_size=3, padding=1),
            get_norm_layer(widths[-1], widths[-1], n_groups, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv3d(widths[-1], widths[-1], kernel_size=3, padding=1),
            get_norm_layer(widths[-1], widths[-1], n_groups, norm_type),
            nn.ReLU(inplace=True),
        )
        self.final_mamba_block = MambaBlock(widths[-1], d_state=d_state, expand=expand)

    def forward(self, x):  # [B, C, T, H, W]
        skips = []
        for conv, mamba in zip(self.layers, self.mamba_blocks):
            x = conv(x)
            x_tavg = x.mean(dim=2)
            x = mamba(x_tavg) + x_tavg
            skips.append(x)

        x = self.final_encoder_conv3d(x.unsqueeze(2))
        x_tavg = x.mean(dim=2)
        x = self.final_mamba_block(x_tavg) + x_tavg
        skips.append(x)

        return x, skips


class Decoder(nn.Module):
    def __init__(self, decoder_widths, out_channels, covmode='diag', norm_type='group', n_groups=4):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        self.skip_connection = True

        for i in range(len(decoder_widths)):
            in_c = decoder_widths[i - 1] if i > 0 else decoder_widths[0]
            out_c = decoder_widths[i]
            self.decoder_blocks.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                get_norm_layer(out_c, out_c, n_groups, norm_type),
                nn.ReLU(inplace=True)
            ))

        self.recon_head = nn.Conv2d(decoder_widths[-1], out_channels, 1)
        if covmode == 'diag':
            self.uncertainty_head = nn.Conv2d(decoder_widths[-1], out_channels, 1)
        elif covmode == 'iso':
            self.uncertainty_head = nn.Conv2d(decoder_widths[-1], 1, 1)
        elif covmode == 'uni':
            self.uncertainty_head = nn.Conv2d(decoder_widths[-1], out_channels, 1)
        elif covmode == 'full':
            self.uncertainty_head = nn.Conv2d(decoder_widths[-1], out_channels * out_channels, 1)
        else:
            self.uncertainty_head = nn.Identity()

    def forward(self, x, skips):
        for i, block in enumerate(self.decoder_blocks):
            if self.skip_connection and i < len(skips):
                x = x + skips[-(i + 1)]
            x = block(x)
        return self.recon_head(x), self.uncertainty_head(x)

def get_norm_layer(channels, num_feats, n_groups=4, layer_type='group'):
    if layer_type == 'batch':
        return nn.BatchNorm2d(channels)
    elif layer_type == 'instance':
        return nn.InstanceNorm2d(channels)
    elif layer_type == 'group':
        return nn.GroupNorm(num_channels=num_feats, num_groups=n_groups)
    else:
        return nn.Identity()


def get_nonlinearity(mode, eps=1e-6):
    if mode == 'relu':
        return lambda x: F.relu(x) + eps
    elif mode == 'softplus':
        return lambda x: F.softplus(x) + eps
    elif mode == 'elu':
        return lambda x: F.elu(x) + 1 + eps
    else:
        return nn.Identity()

class UNCRTAINTS_Mamba3D(nn.Module):
    def __init__(self, input_dim, output_dim=13,
                 encoder_widths=[128, 128],
                 decoder_widths=[128, 128],
                 d_state=16, covmode='diag', scale_by=1.0,
                 out_nonlin_mean=False, out_nonlin_var='relu',
                 norm_type='group'):
        super().__init__()
        self.mean_idx = output_dim
        self.scale_by = scale_by
        self.covmode = covmode

        self.input_projection = nn.Conv3d(input_dim, encoder_widths[0], kernel_size=3, padding=1)

        self.pre_encoder_conv3d = nn.Sequential(
            nn.Conv3d(encoder_widths[0], encoder_widths[0], kernel_size=3, padding=1),
            get_norm_layer(encoder_widths[0], encoder_widths[0], n_groups=4, layer_type=norm_type),
            nn.ReLU(inplace=True),
            nn.Conv3d(encoder_widths[0], encoder_widths[0], kernel_size=3, padding=1),
            get_norm_layer(encoder_widths[0], encoder_widths[0], n_groups=4, layer_type=norm_type),
            nn.ReLU(inplace=True),
        )

        self.encoder = TemporalEncoder(
            widths=encoder_widths,
            d_state=d_state,
            expand=2.0,
            norm_type=norm_type
        )

        self.decoder = Decoder(
            decoder_widths=decoder_widths,
            out_channels=output_dim,
            covmode=covmode,
            norm_type=norm_type
        )

        self.out_mean = nn.Sigmoid() if out_nonlin_mean else nn.Identity()
        self.nonlin_var = get_nonlinearity(out_nonlin_var, eps=1e-6)

    def forward(self, x, batch_positions=None):  # [B, T, C, H, W]
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.input_projection(x)
        x = self.pre_encoder_conv3d(x)
        feat, skips = self.encoder(x)
        recon, uncertainty = self.decoder(feat, skips)
        out_loc = self.scale_by * self.out_mean(recon)

        if self.covmode == 'iso':
            out_cov = self.nonlin_var(uncertainty).expand(-1, self.mean_idx, -1, -1)
        elif self.covmode in ['diag', 'uni', 'full']:
            out_cov = self.nonlin_var(uncertainty)
        else:
            out_cov = torch.zeros_like(out_loc)

        return torch.cat([out_loc, out_cov], dim=1).unsqueeze(1)




































































