from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -scale)
        emb = t[:, None].float() * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        groups = min(8, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = h + self.time_projection(F.gelu(t_emb))[:, :, None, None]
        h = F.gelu(self.norm2(self.conv2(h)))
        return h + self.residual(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attention = torch.softmax(q.transpose(-2, -1) @ k / math.sqrt(self.head_dim), dim=-1)
        out = (v @ attention.transpose(-2, -1)).reshape(b, c, h, w)
        return self.out(out) + x


class CrossAttention(nn.Module):
    def __init__(self, channels: int, context_dim: int, heads: int):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.context_norm = nn.LayerNorm(context_dim)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Linear(context_dim, channels * 2)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(self.norm(x)).reshape(b, self.heads, self.head_dim, h * w)
        kv = self.kv(self.context_norm(context)).reshape(b, 2, self.heads, self.head_dim)
        k, v = kv[:, 0], kv[:, 1]
        k = k[:, :, :, None].expand(-1, -1, -1, h * w)
        v = v[:, :, :, None].expand(-1, -1, -1, h * w)
        attention = torch.softmax(q.transpose(-2, -1) @ k / math.sqrt(self.head_dim), dim=-1)
        out = (v @ attention.transpose(-2, -1)).reshape(b, c, h, w)
        return self.out(out) + x


class SpectralDiffusionUNet(nn.Module):
    """Paper-style spectral diffusion U-Net for conditional 1D spectra generation.

    A 1D Raman spectrum is padded and reshaped to a 2D matrix so that convolution,
    self-attention, and label cross-attention can model local peaks and global
    spectral structure before the predicted noise is flattened back to 1D.
    """

    def __init__(
        self,
        spec_len: int = 1784,
        time_emb_dim: int = 128,
        num_classes: int = 2,
        label_dim: int = 128,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] | list[int] = (1, 2, 4),
        num_heads: int = 4,
        dropout: float = 0.0,
        reshape_height: int | None = None,
        reshape_width: int | None = None,
    ):
        super().__init__()
        self.spec_len = spec_len
        if reshape_height and reshape_width and reshape_height * reshape_width >= spec_len:
            self.height = int(reshape_height)
            self.width = int(reshape_width)
        else:
            side = math.ceil(math.sqrt(spec_len))
            self.height = side
            self.width = side
        self.padded_len = self.height * self.width
        self.dropout = nn.Dropout(dropout)

        channels = [base_channels * multiplier for multiplier in channel_multipliers]
        if len(channels) < 2:
            raise ValueError("channel_multipliers must contain at least two levels")

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.label_mlp = nn.Sequential(
            nn.Linear(num_classes, label_dim),
            nn.GELU(),
            nn.Linear(label_dim, label_dim),
        )

        self.enc_blocks = nn.ModuleList()
        in_channels = 1
        for out_channels in channels:
            self.enc_blocks.append(Block(in_channels, out_channels, time_emb_dim))
            in_channels = out_channels

        self.pool = nn.MaxPool2d(2)
        bottleneck_channels = channels[-1]
        self.bot1 = Block(bottleneck_channels, bottleneck_channels, time_emb_dim)
        self.bot_sa = SelfAttention(bottleneck_channels, num_heads)
        self.bot_time_ca = CrossAttention(bottleneck_channels, time_emb_dim, num_heads)
        self.bot_label_ca = CrossAttention(bottleneck_channels, label_dim, num_heads)
        self.bot2 = Block(bottleneck_channels, bottleneck_channels, time_emb_dim)

        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        current = reversed_channels[0]
        for skip_channels in reversed_channels[1:]:
            self.up_blocks.append(
                nn.ModuleDict(
                    {
                        "up": nn.ConvTranspose2d(current, skip_channels, 2, stride=2),
                        "block": Block(skip_channels * 2, skip_channels, time_emb_dim),
                        "label_ca": CrossAttention(skip_channels, label_dim, num_heads),
                    }
                )
            )
            current = skip_channels

        self.out_conv = nn.Conv2d(channels[0], 1, 1)

    def _pad_and_reshape(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.padded_len:
            x = F.pad(x, (0, self.padded_len - x.shape[1]))
        return x.reshape(x.shape[0], 1, self.height, self.width)

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)[:, : self.spec_len]

    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        label_emb = self.label_mlp(label)
        x = self._pad_and_reshape(x)

        skips = []
        for index, block in enumerate(self.enc_blocks):
            x = block(x, t_emb)
            skips.append(x)
            if index != len(self.enc_blocks) - 1:
                x = self.pool(x)

        x = self.bot1(x, t_emb)
        x = self.bot_sa(x)
        x = self.bot_time_ca(x, t_emb)
        x = self.bot_label_ca(x, label_emb)
        x = self.bot2(self.dropout(x), t_emb)

        decoder_skips = list(reversed(skips[:-1]))
        for module, skip in zip(self.up_blocks, decoder_skips):
            x = module["up"](x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = module["block"](x, t_emb)
            x = module["label_ca"](x, label_emb)

        return self._flatten(self.out_conv(x))
