"""Frozen ResNet teacher feature extractor."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from iqa.models.common.architectures import build_resnet
from iqa.models.feature_ae.models import DEFAULT_FEATURE_LAYERS, normalize_feature_layers


class ResNetTeacherFeatures(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        layers: Iterable[str] = DEFAULT_FEATURE_LAYERS,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.layers = normalize_feature_layers(layers)
        self.model = build_resnet(backbone, pretrained=pretrained).eval()
        self._features: dict[str, torch.Tensor] = {}
        modules = dict(self.model.named_modules())
        for layer_name in self.layers:
            modules[layer_name].register_forward_hook(self._make_hook(layer_name))
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def _make_hook(self, layer_name: str):
        def hook(_module, _inputs, output):
            self._features[layer_name] = output.detach()

        return hook

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._features = {}
        with torch.no_grad():
            _ = self.model(x)
        return {name: self._features[name].detach() for name in self.layers}


__all__ = ["ResNetTeacherFeatures"]
