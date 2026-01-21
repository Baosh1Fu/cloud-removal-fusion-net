"""
UnCRtainTS Implementation
Author: Patrick Ebel (github/patrickTUM)
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.backbones.utae import ConvLayer, ConvBlock, TemporallySharedBlock
from src.backbones.ltae import LTAE2d, LTAE2dtiny

S2_BANDS = 13


def get_norm_layer(out_channels, num_feats, n_groups=4, layer_type='batch'):
    if layer_type == 'batch':
        return nn.BatchNorm2d(out_channels)
    elif layer_type == 'instance':
        return nn.InstanceNorm2d(out_channels)
    elif layer_type == 'group':
        return nn.GroupNorm(num_channels=num_feats, num_groups=n_groups)

class ResidualConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        n_groups=4,
        #last_relu=True,
        k=3, s=1, p=1,
        padding_mode="reflect",
    ):
        super(ResidualConvBlock, self).__init__(pad_value=pad_value)

        self.conv1 = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=True,
            k=k, s=s, p=p,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=True,
            k=k, s=s, p=p,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )
        self.conv3 = ConvLayer(
            nkernels=nkernels,
            #norm='none',
            #last_relu=False,
            norm=norm,
            last_relu=True,
            k=k, s=s, p=p,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )

    def forward(self, input):

        out1 = self.conv1(input)        # followed by built-in ReLU & norm
        out2 = self.conv2(out1)         # followed by built-in ReLU & norm
        out3 = input + self.conv3(out2) # omit norm & ReLU
        return out3


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm, n_groups=4):
        super().__init__()
        self.norm = get_norm_layer(dim, dim, n_groups, norm)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

    
class MBConv(TemporallySharedBlock):
    def __init__(self, inp, oup, downsample=False, expansion=4, norm='batch', n_groups=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                          padding=1, padding_mode='reflect', groups=hidden_dim, bias=False),
                get_norm_layer(hidden_dim, hidden_dim, n_groups, norm),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, stride=1, padding=0, bias=False),
                get_norm_layer(oup, oup, n_groups, norm),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride=stride, padding=0, bias=False),
                get_norm_layer(hidden_dim, hidden_dim, n_groups, norm),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, padding_mode='reflect',
                          groups=hidden_dim, bias=False),
                get_norm_layer(hidden_dim, hidden_dim, n_groups, norm),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, stride=1, padding=0, bias=False),
                get_norm_layer(oup, oup, n_groups, norm), 
            )
        
        self.conv = PreNorm(inp, self.conv, norm, n_groups=4)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Compact_Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Compact_Temporal_Aggregator, self).__init__()
        self.mode = mode
        # moved dropout from ScaledDotProductAttention to here, applied after upsampling 
        self.attn_dropout = nn.Dropout(0.1) # no dropout via: nn.Dropout(0.0)

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                    # this got moved out of ScaledDotProductAttention, apply after upsampling
                    attn = self.attn_dropout(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                # this got moved out of ScaledDotProductAttention, apply after upsampling
                attn = self.attn_dropout(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                    # this got moved out of ScaledDotProductAttention, apply after upsampling
                    attn = self.attn_dropout(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                # this got moved out of ScaledDotProductAttention, apply after upsampling
                attn = self.attn_dropout(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)

def get_nonlinearity(mode, eps):
    if mode=='relu':        fct = nn.ReLU() + eps 
    elif mode=='softplus':  fct = lambda vars:nn.Softplus(beta=1, threshold=20)(vars) + eps
    elif mode=='elu':       fct = lambda vars: nn.ELU()(vars) + 1 + eps  
    else:                   fct = nn.Identity()
    return fct



# Global constant: number of optical bands
S2_BANDS = 13
class NIGHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.delta = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gamma = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.alpha = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.beta  = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, B, T, H, W):
        delta = self.delta(x)
        gamma = F.softplus(self.gamma(x)) + self.eps
        alpha = F.softplus(self.alpha(x)) + 1.0
        beta  = F.softplus(self.beta(x)) + self.eps

        return {
            'delta': delta.view(B, 1, -1, H, W),
            'gamma': gamma.view(B, 1, -1, H, W),
            'alpha': alpha.view(B, 1, -1, H, W),
            'beta':  beta.view(B, 1, -1, H, W),
        }

S2_BANDS = 13

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

class UNCRTAINTS(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[128],
        decoder_widths=[128, 128, 128, 128, 128],
        out_nonlin_mean=False,
        out_nonlin_var='relu',
        agg_mode="att_group",
        encoder_norm="group",
        decoder_norm="batch",
        n_head=16,
        d_model=256,
        d_k=4,
        pad_value=0,
        positional_encoding=True,
        use_v=False,
        block_type='mbconv',
        is_mono=False
    ):
        super().__init__()

        self.pad_value = pad_value
        self.is_mono = is_mono
        self.use_v = use_v

        # Spatial encoder
        self.in_conv = ConvBlock([input_dim, encoder_widths[0]], k=1, s=1, p=0, norm=encoder_norm)
        self.in_block = nn.ModuleList([
            MBConv(c, c, downsample=False, expansion=2, norm=encoder_norm)
            for c in encoder_widths
        ])

        # Temporal encoder
        if not is_mono:
            if use_v:
                self.temporal_encoder = LTAE2d(
                    in_channels=encoder_widths[0], d_model=d_model, n_head=n_head,
                    mlp=[d_model, encoder_widths[0]], return_att=True, d_k=d_k,
                    positional_encoding=positional_encoding, use_dropout=False
                )
                self.include_v = nn.Conv2d(encoder_widths[0]*2, encoder_widths[0], kernel_size=1)
            else:
                self.temporal_encoder = LTAE2dtiny(
                    in_channels=encoder_widths[0], d_model=d_model, n_head=n_head, d_k=d_k,
                    positional_encoding=positional_encoding
                )
            self.temporal_aggregator = Compact_Temporal_Aggregator(mode=agg_mode)

        # Spatial decoder
        self.out_block = nn.ModuleList([
            MBConv(c, c, downsample=False, expansion=2, norm=decoder_norm)
            for c in decoder_widths
        ])

        # Only one NIG output head at the end
        self.main_nig_head = NIGHead(decoder_widths[0], S2_BANDS)

    def forward(self, x, batch_positions=None):
        B, T, C, H, W = x.shape
        pad_mask = (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)

        out = x.view(B*T, C, H, W)
        out = self.in_conv.smart_forward(out)
        for layer in self.in_block:
            out = layer.smart_forward(out)
        out = out.view(B, T, -1, H, W)

        if not self.is_mono:
            att_down = 32
            down = F.adaptive_max_pool2d(out.view(B*T, -1, H, W), (att_down, att_down)).view(B, T, -1, att_down, att_down)
            if self.use_v:
                v, att = self.temporal_encoder(down, batch_positions=batch_positions, pad_mask=pad_mask)
            else:
                att = self.temporal_encoder(down, batch_positions=batch_positions, pad_mask=pad_mask)
            out = self.temporal_aggregator(out, pad_mask=pad_mask, attn_mask=att)
            if self.use_v:
                up_v = F.interpolate(v.view(B*T, -1, att_down, att_down), size=(H, W), mode="bilinear", align_corners=False).view(B, T, -1, H, W)
                out = self.include_v(torch.cat([out, up_v], dim=2))
        else:
            out = out.squeeze(1)

        if out.dim() == 5:
            out = out.view(B*T, out.size(2), H, W)

        for layer in self.out_block:
            out = layer.smart_forward(out)

        main_nig_out = self.main_nig_head(out, B, T, H, W)

        return {
            'delta': main_nig_out['delta'],
            'gamma': main_nig_out['gamma'],
            'alpha': main_nig_out['alpha'],
            'beta':  main_nig_out['beta'],
        }