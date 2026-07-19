from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import SegformerForSemanticSegmentation


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * torch.empty(shape, device=x.device).bernoulli_(keep) / keep


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, dim: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, dim, kernel, stride, padding=kernel // 2
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.projection(x)
        height, width = x.shape[-2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, height, width


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, reduction: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Embedding dimension must be divisible by head count")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, dim * 2)
        self.output = nn.Linear(dim, dim)
        self.reduction = reduction
        if reduction > 1:
            self.sr = nn.Conv2d(dim, dim, reduction, reduction)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        query = (
            self.query(x)
            .reshape(batch, tokens, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        source = x
        if self.reduction > 1:
            source = x.transpose(1, 2).reshape(batch, channels, height, width)
            source = self.sr(source).flatten(2).transpose(1, 2)
            source = self.norm(source)
        key, value = self.key_value(source).chunk(2, dim=-1)
        key = key.reshape(batch, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        value = (
            value.reshape(batch, -1, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        attention = (query @ key.transpose(-2, -1) * self.scale).softmax(dim=-1)
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.output(x)


class MixFFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = dim * expansion
        self.first = nn.Linear(dim, hidden)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.activation = nn.GELU()
        self.second = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch = x.shape[0]
        x = self.first(x)
        x = x.transpose(1, 2).reshape(batch, -1, height, width)
        x = self.depthwise(x).flatten(2).transpose(1, 2)
        return self.second(self.activation(x))


class TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, heads: int, reduction: int, drop_path: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = EfficientSelfAttention(dim, heads, reduction)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x), height, width))
        x = x + self.drop_path(self.ffn(self.norm2(x), height, width))
        return x


class MiTEncoder(nn.Module):
    """Mix Transformer B0 encoder adapted to four input bands."""

    def __init__(self, in_channels: int = 4, drop_path: float = 0.1) -> None:
        super().__init__()
        dims = (32, 64, 160, 256)
        heads = (1, 2, 5, 8)
        depths = (2, 2, 2, 2)
        reductions = (8, 4, 2, 1)
        total_blocks = sum(depths)
        rates = torch.linspace(0, drop_path, total_blocks).tolist()
        self.embeddings = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        block_index = 0
        channels = in_channels
        for stage, (dim, head, depth, reduction) in enumerate(
            zip(dims, heads, depths, reductions)
        ):
            kernel, stride = (7, 4) if stage == 0 else (3, 2)
            self.embeddings.append(
                OverlapPatchEmbedding(channels, dim, kernel, stride)
            )
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(
                    TransformerBlock(dim, head, reduction, rates[block_index])
                )
                block_index += 1
            self.blocks.append(nn.ModuleList(stage_blocks))
            self.norms.append(nn.LayerNorm(dim))
            channels = dim

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for embedding, blocks, norm in zip(
            self.embeddings, self.blocks, self.norms
        ):
            x, height, width = embedding(x)
            for block in blocks:
                x = block(x, height, width)
            x = norm(x)
            x = x.transpose(1, 2).reshape(x.shape[0], -1, height, width)
            features.append(x)
        return features


class SegFormerHead(nn.Module):
    def __init__(self, encoder_dims: tuple[int, ...], decoder_dim: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(dim, decoder_dim, 1) for dim in encoder_dims
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim, 1, 1),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projected = [
            F.interpolate(
                projection(feature),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            for projection, feature in zip(self.projections, features)
        ]
        return self.fuse(torch.cat(projected, dim=1))


class SegFormer(nn.Module):
    """SegFormer-B0 binary segmenter with a four-band RGBN input stem."""

    def __init__(self, in_channels: int = 4, decoder_dim: int = 128) -> None:
        super().__init__()
        self.encoder = MiTEncoder(in_channels)
        self.decoder = SegFormerHead((32, 64, 160, 256), decoder_dim)
        self.apply(self._initialize)
        classifier = self.decoder.fuse[-1]
        nn.init.trunc_normal_(classifier.weight, std=0.02)
        nn.init.zeros_(classifier.bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1]
            fan_out *= module.out_channels / module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        logits = self.decoder(self.encoder(x))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


class PretrainedSegFormer(nn.Module):
    """NVIDIA SegFormer-B0 with pretrained RGB weights expanded to RGBN."""

    def __init__(
        self,
        pretrained_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.network = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            num_labels=1,
            id2label={0: "road"},
            label2id={"road": 0},
            ignore_mismatched_sizes=True,
            local_files_only=local_files_only,
        )
        embedding = self.network.segformer.encoder.patch_embeddings[0]
        original = embedding.proj
        replacement = nn.Conv2d(
            4,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )
        with torch.no_grad():
            replacement.weight[:, :3].copy_(original.weight)
            replacement.weight[:, 3:4].copy_(original.weight.mean(dim=1, keepdim=True))
            if original.bias is not None:
                replacement.bias.copy_(original.bias)
        embedding.proj = replacement

    @property
    def encoder_parameters(self):
        return self.network.segformer.parameters()

    @property
    def head_parameters(self):
        return self.network.decode_head.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        logits = self.network(pixel_values=x).logits
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def build_model(config: dict) -> nn.Module:
    if "pretrained_name" in config:
        return PretrainedSegFormer(**config)
    return SegFormer(**config)
