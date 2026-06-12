"""Shared runtime building blocks for IQA models."""

from __future__ import annotations

import warnings

from torch import nn
from torchvision import models


def conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def build_resnet(backbone: str = "resnet18", *, pretrained: bool = True) -> nn.Module:
    if backbone != "resnet18":
        raise ValueError(f"Unsupported backbone {backbone!r}.")

    if pretrained:
        try:
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:
            warnings.warn(
                f"Could not load torchvision ResNet18 weights ({exc}). Falling back to random weights.",
                RuntimeWarning,
                stacklevel=2,
            )
    return models.resnet18(weights=None)


__all__ = ["build_resnet", "conv_block"]
