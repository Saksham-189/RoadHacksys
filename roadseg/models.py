from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x), inplace=True)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class BasicUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.e1 = ConvBlock(in_channels, c)
        self.e2 = ConvBlock(c, c * 2)
        self.e3 = ConvBlock(c * 2, c * 4)
        self.e4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 16)
        self.d4 = UpBlock(c * 16, c * 8, c * 8)
        self.d3 = UpBlock(c * 8, c * 4, c * 4)
        self.d2 = UpBlock(c * 4, c * 2, c * 2)
        self.d1 = UpBlock(c * 2, c, c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d = self.d4(b, e4)
        d = self.d3(d, e3)
        d = self.d2(d, e2)
        d = self.d1(d, e1)
        return self.head(d)


class ResNetUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.stem = ResidualBlock(in_channels, c)
        self.e2 = ResidualBlock(c, c * 2, stride=2)
        self.e3 = ResidualBlock(c * 2, c * 4, stride=2)
        self.e4 = ResidualBlock(c * 4, c * 8, stride=2)
        self.bottleneck = ResidualBlock(c * 8, c * 16, stride=2)
        self.d4 = UpBlock(c * 16, c * 8, c * 8)
        self.d3 = UpBlock(c * 8, c * 4, c * 4)
        self.d2 = UpBlock(c * 4, c * 2, c * 2)
        self.d1 = UpBlock(c * 2, c, c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.stem(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b = self.bottleneck(e4)
        d = self.d4(b, e4)
        d = self.d3(d, e3)
        d = self.d2(d, e2)
        d = self.d1(d, e1)
        return self.head(d)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.pool = nn.MaxPool2d(2)
        self.x00 = ConvBlock(in_channels, c)
        self.x10 = ConvBlock(c, c * 2)
        self.x20 = ConvBlock(c * 2, c * 4)
        self.x30 = ConvBlock(c * 4, c * 8)
        self.x01 = ConvBlock(c + c * 2, c)
        self.x11 = ConvBlock(c * 2 + c * 4, c * 2)
        self.x21 = ConvBlock(c * 4 + c * 8, c * 4)
        self.x02 = ConvBlock(c * 2 + c * 2, c)
        self.x12 = ConvBlock(c * 4 + c * 4, c * 2)
        self.x03 = ConvBlock(c * 3 + c * 2, c)
        self.head = nn.Conv2d(c, 1, 1)

    @staticmethod
    def up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x01 = self.x01(torch.cat([x00, self.up(x10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, self.up(x20, x10)], dim=1))
        x21 = self.x21(torch.cat([x20, self.up(x30, x20)], dim=1))
        x02 = self.x02(torch.cat([x00, x01, self.up(x11, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21, x10)], dim=1))
        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12, x00)], dim=1))
        return self.head(x03)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_ch, out_ch, 1),
                nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
                nn.Conv2d(in_ch, out_ch, 3, padding=4, dilation=4),
                nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6),
            ]
        )
        self.project = ConvBlock(out_ch * 4, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.e1 = ResidualBlock(in_channels, c)
        self.e2 = ResidualBlock(c, c * 2, stride=2)
        self.e3 = ResidualBlock(c * 2, c * 4, stride=2)
        self.e4 = ResidualBlock(c * 4, c * 8, stride=2)
        self.aspp = ASPP(c * 8, c * 4)
        self.low = nn.Conv2d(c, c, 1)
        self.decoder = ConvBlock(c * 5, c * 2)
        self.head = nn.Conv2d(c * 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        y = self.aspp(e4)
        y = F.interpolate(y, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.decoder(torch.cat([y, self.low(e1)], dim=1))
        y = self.head(y)
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class PatchTransformerSeg(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.decoder = nn.Sequential(
            ConvBlock(embed_dim, embed_dim // 2),
            nn.Conv2d(embed_dim // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feat = self.patch(x)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        feat = tokens.transpose(1, 2).reshape(b, c, h, w)
        out = self.decoder(feat)
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class TransUNetLite(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, embed_dim: int = 128) -> None:
        super().__init__()
        c = base_channels
        self.e1 = ConvBlock(in_channels, c)
        self.e2 = ConvBlock(c, c * 2)
        self.e3 = ConvBlock(c * 2, embed_dim)
        self.pool = nn.MaxPool2d(2)
        layer = nn.TransformerEncoderLayer(embed_dim, 4, embed_dim * 4, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.d2 = UpBlock(embed_dim, c * 2, c * 2)
        self.d1 = UpBlock(c * 2, c, c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b, c, h, w = e3.shape
        tokens = self.transformer(e3.flatten(2).transpose(1, 2))
        y = tokens.transpose(1, 2).reshape(b, c, h, w)
        y = self.d2(y, e2)
        y = self.d1(y, e1)
        return self.head(y)


class SwinUNetLite(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, embed_dim: int = 96, patch_size: int = 8) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, base_channels)
        self.segformer = PatchTransformerSeg(base_channels, embed_dim=embed_dim, depth=3, num_heads=4, patch_size=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segformer(self.stem(x))


class Mask2FormerLite(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 96, num_queries: int = 8, patch_size: int = 8) -> None:
        super().__init__()
        self.encoder = PatchTransformerSeg(in_channels, embed_dim=embed_dim, depth=3, num_heads=4, patch_size=patch_size)
        self.pixel = ConvBlock(1, embed_dim // 2)
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim // 2) * 0.02)
        self.query_score = nn.Linear(embed_dim // 2, 1)
        self.out = nn.Conv2d(num_queries, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse = self.encoder(x)
        pixel = self.pixel(coarse)
        queries = self.query_embed
        masks = torch.einsum("bchw,qc->bqhw", pixel, queries)
        scores = torch.sigmoid(self.query_score(queries)).view(1, -1, 1, 1)
        return self.out(masks * scores)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    backbone: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "unet": ModelSpec("U-Net", "CNN", "Basic CNN"),
    "resnet_unet": ModelSpec("ResNet U-Net", "CNN", "Residual encoder"),
    "unetpp": ModelSpec("UNet++", "CNN", "Nested skips"),
    "deeplabv3plus": ModelSpec("DeepLabV3+", "CNN", "Residual ASPP"),
    "segformer": ModelSpec("SegFormer", "Transformer", "MiT-style patch transformer"),
    "swin_unet": ModelSpec("Swin-Unet", "Transformer", "Swin-style lightweight encoder"),
    "transunet": ModelSpec("TransUNet", "Transformer", "CNN + ViT hybrid"),
    "mask2former": ModelSpec("Mask2Former", "Transformer", "Query-mask decoder"),
    "dinov2_vit_head": ModelSpec("DINO/ViT Head", "Transformer", "ViT segmentation head"),
}


def build_model(model_cfg: dict[str, Any], in_channels: int) -> nn.Module:
    name = str(model_cfg["name"]).lower()
    base = int(model_cfg.get("base_channels", 32))
    embed = int(model_cfg.get("embed_dim", 96))
    patch_size = int(model_cfg.get("patch_size", 8))

    if name == "unet":
        return BasicUNet(in_channels, base)
    if name == "resnet_unet":
        return ResNetUNet(in_channels, base)
    if name == "unetpp":
        return UNetPlusPlus(in_channels, base)
    if name == "deeplabv3plus":
        return DeepLabV3Plus(in_channels, base)
    if name == "segformer":
        return PatchTransformerSeg(in_channels, embed_dim=embed, depth=int(model_cfg.get("depth", 4)), patch_size=patch_size)
    if name == "swin_unet":
        return SwinUNetLite(in_channels, base_channels=base, embed_dim=embed, patch_size=patch_size)
    if name == "transunet":
        return TransUNetLite(in_channels, base_channels=base, embed_dim=max(embed, 64))
    if name == "mask2former":
        return Mask2FormerLite(
            in_channels,
            embed_dim=embed,
            num_queries=int(model_cfg.get("num_queries", 8)),
            patch_size=patch_size,
        )
    if name == "dinov2_vit_head":
        return PatchTransformerSeg(in_channels, embed_dim=embed, depth=int(model_cfg.get("depth", 6)), patch_size=patch_size)
    raise ValueError(f"Unsupported model: {name}")


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000


def model_spec(name: str) -> ModelSpec:
    return MODEL_SPECS[name.lower()]
