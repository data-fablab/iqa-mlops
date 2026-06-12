"""RD Feature-AE retained runtime architecture."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from iqa.models.common.architectures import conv_block


FEATURE_AE_MODEL_TYPE = "reverse_distill_resnet18_dual_context_gated"
SUPPORTED_TEACHER_BACKBONE = "resnet18"
DEFAULT_FEATURE_LAYERS = ("layer2", "layer3")
TEACHER_LAYER_CHANNELS = {"layer1": 64, "layer2": 128, "layer3": 256}


def normalize_feature_layers(layers: Iterable[str] = DEFAULT_FEATURE_LAYERS) -> tuple[str, ...]:
    normalized = tuple(layers)
    unknown = [layer for layer in normalized if layer not in TEACHER_LAYER_CHANNELS]
    if unknown:
        raise ValueError(f"Unknown teacher feature layer(s): {', '.join(unknown)}.")
    return normalized


class ReverseDistillationGatedDualContextResNet18(nn.Module):
    """Retained gated reverse-distillation Feature-AE runtime."""

    model_type = FEATURE_AE_MODEL_TYPE
    teacher_backbone = SUPPORTED_TEACHER_BACKBONE
    stem_channels = 64
    layer1_channels = 64
    layer2_base_channels = 128
    layer3_base_channels = 256

    def __init__(self, layers: Iterable[str] = DEFAULT_FEATURE_LAYERS) -> None:
        super().__init__()
        self.layers = normalize_feature_layers(layers)
        c0 = int(self.stem_channels)
        c1 = int(self.layer1_channels)
        c2 = int(self.layer2_base_channels)
        c3 = int(self.layer3_base_channels)

        self.stem = nn.Sequential(conv_block(3, c0, stride=2), conv_block(c0, c0))
        self.down1 = nn.Sequential(conv_block(c0, c1, stride=2), conv_block(c1, c1))
        self.layer1_block = nn.Sequential(conv_block(c1, c1), conv_block(c1, c1))
        self.down2 = nn.Sequential(conv_block(c1, c2, stride=2), conv_block(c2, c2))
        self.layer2_head = nn.Sequential(conv_block(c2, c2), nn.Conv2d(c2, TEACHER_LAYER_CHANNELS["layer2"], 1))
        self.down3 = nn.Sequential(conv_block(c2, c3, stride=2), conv_block(c3, c3))
        self.layer3_head = nn.Sequential(conv_block(c3, c3), nn.Conv2d(c3, TEACHER_LAYER_CHANNELS["layer3"], 1))
        self.layer1_head = nn.Sequential(conv_block(c1, c1), nn.Conv2d(c1, TEACHER_LAYER_CHANNELS["layer1"], 1))

        self.encoder_stem = nn.Sequential(conv_block(3, c0, stride=2), conv_block(c0, c0))
        self.encoder_down1 = nn.Sequential(conv_block(c0, c1, stride=2), conv_block(c1, c1))
        self.encoder_down2 = nn.Sequential(conv_block(c1, c2, stride=2), conv_block(c2, c2))
        self.encoder_down3 = nn.Sequential(conv_block(c2, c3, stride=2), conv_block(c3, c3))
        self.bottleneck = nn.Sequential(
            conv_block(c3, c3),
            nn.Conv2d(c3, c3, kernel_size=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            conv_block(c3, c3),
        )

        self.decode_layer3 = nn.Sequential(conv_block(c3, c3), nn.Conv2d(c3, TEACHER_LAYER_CHANNELS["layer3"], 1))
        self.up_layer2 = nn.Sequential(conv_block(c3, c2), conv_block(c2, c2))
        self.decode_layer2 = nn.Sequential(conv_block(c2, c2), nn.Conv2d(c2, TEACHER_LAYER_CHANNELS["layer2"], 1))
        self.up_layer1 = nn.Sequential(conv_block(c2, c1), conv_block(c1, c1))
        self.decode_layer1 = nn.Sequential(conv_block(c1, c1), nn.Conv2d(c1, TEACHER_LAYER_CHANNELS["layer1"], 1))

        self.context_stem = nn.Sequential(conv_block(3, c0, stride=2), conv_block(c0, c0))
        self.context_down1 = nn.Sequential(conv_block(c0, c1, stride=2), conv_block(c1, c1))
        self.context_down2 = nn.Sequential(conv_block(c1, c2, stride=2), conv_block(c2, c2))
        self.context_down3 = nn.Sequential(conv_block(c2, c3, stride=2), conv_block(c3, c3))
        self.context_film = nn.Conv2d(c3, 2 * c3, kernel_size=1)
        self.context_gate = nn.Conv2d(c3, c3, kernel_size=1)
        nn.init.zeros_(self.context_film.weight)
        nn.init.zeros_(self.context_film.bias)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.constant_(self.context_gate.bias, -2.0)

    def encode_local(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder_stem(images)
        x = self.encoder_down1(x)
        x = self.encoder_down2(x)
        x = self.encoder_down3(x)
        return self.bottleneck(x)

    def decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        if "layer3" in self.layers:
            outputs["layer3"] = self.decode_layer3(latent)

        layer2_latent = F.interpolate(latent, scale_factor=2.0, mode="bilinear", align_corners=False)
        layer2_latent = self.up_layer2(layer2_latent)
        if "layer2" in self.layers:
            outputs["layer2"] = self.decode_layer2(layer2_latent)

        layer1_latent = F.interpolate(layer2_latent, scale_factor=2.0, mode="bilinear", align_corners=False)
        layer1_latent = self.up_layer1(layer1_latent)
        if "layer1" in self.layers:
            outputs["layer1"] = self.decode_layer1(layer1_latent)
        return outputs

    def encode_context(self, context_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.context_stem(context_images)
        context = self.context_down1(context)
        context = self.context_down2(context)
        context = self.context_down3(context)
        gamma_beta = self.context_film(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gate = torch.sigmoid(self.context_gate(context))
        return 0.25 * torch.tanh(gamma), 0.25 * torch.tanh(beta), gate

    def forward(self, images: torch.Tensor, context_images: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if context_images is None:
            context_images = images
        latent = self.encode_local(images)
        gamma, beta, gate = self.encode_context(context_images)
        if gamma.shape[-2:] != latent.shape[-2:]:
            gamma = F.interpolate(gamma, size=latent.shape[-2:], mode="bilinear", align_corners=False)
            beta = F.interpolate(beta, size=latent.shape[-2:], mode="bilinear", align_corners=False)
            gate = F.interpolate(gate, size=latent.shape[-2:], mode="bilinear", align_corners=False)
        latent = latent + gate * (latent * gamma + beta)
        return self.decode(latent)


__all__ = [
    "DEFAULT_FEATURE_LAYERS",
    "FEATURE_AE_MODEL_TYPE",
    "SUPPORTED_TEACHER_BACKBONE",
    "TEACHER_LAYER_CHANNELS",
    "ReverseDistillationGatedDualContextResNet18",
    "normalize_feature_layers",
]
