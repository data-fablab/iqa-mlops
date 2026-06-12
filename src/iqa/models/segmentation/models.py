"""Fixed ROI segmenter runtime architecture."""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from iqa.models.common.architectures import conv_block


ROI_SEGMENTER_MODEL_TYPE = "functional_unet_resnet18_det1_context2b"


def _resnet18_backbone(pretrained: bool = True) -> nn.Module:
    if pretrained:
        try:
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception as exc:
            warnings.warn(
                f"Could not load torchvision ResNet18 pretrained weights ({exc}). "
                "Falling back to a randomly initialized ResNet18 encoder.",
                RuntimeWarning,
                stacklevel=2,
            )
    return models.resnet18(weights=None)


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if int(in_channels) == conv.in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        channels_to_copy = min(conv.in_channels, int(in_channels))
        new_conv.weight[:, :channels_to_copy] = conv.weight[:, :channels_to_copy]
        if int(in_channels) > conv.in_channels:
            extra = conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, conv.in_channels :] = (
                extra.repeat(1, int(in_channels) - conv.in_channels, 1, 1) * 0.25
            )
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class FunctionalSurfaceUNetResNet18Det1Context2B(nn.Module):
    """Retained two-branch ROI segmenter runtime."""

    model_type = ROI_SEGMENTER_MODEL_TYPE

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        backbone = _resnet18_backbone(pretrained=pretrained)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dec4 = nn.Sequential(conv_block(512, 256), conv_block(256, 256))
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.Sequential(conv_block(256, 128), conv_block(128, 128))
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(conv_block(128, 64), conv_block(64, 64))
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(conv_block(128, 64), conv_block(64, 64))
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec0 = nn.Sequential(conv_block(32, 32), conv_block(32, 32))
        self.out = nn.Conv2d(32, 1, kernel_size=1)

        self.det_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.det_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
        )
        self.objectness = nn.Linear(128, 1)
        self.bbox = nn.Linear(128, 4)

        global_backbone = _resnet18_backbone(pretrained=pretrained)
        global_backbone.conv1 = _adapt_first_conv(global_backbone.conv1, 4)
        self.global_stem = nn.Sequential(global_backbone.conv1, global_backbone.bn1, global_backbone.relu)
        self.global_maxpool = global_backbone.maxpool
        self.global_layer1 = global_backbone.layer1
        self.global_layer2 = global_backbone.layer2
        self.global_layer3 = global_backbone.layer3
        self.global_layer4 = global_backbone.layer4
        self.context_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.context_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )

    def set_encoder_trainable(self, trainable: bool) -> None:
        for module in [
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.global_stem,
            self.global_layer1,
            self.global_layer2,
            self.global_layer3,
            self.global_layer4,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = bool(trainable)

    @staticmethod
    def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = self.stem(images)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        z = self.layer4(x3)
        return x0, x1, x2, x3, z

    def encode_global(self, global_image: torch.Tensor, crop_box_mask: torch.Tensor) -> torch.Tensor:
        if crop_box_mask.ndim == 3:
            crop_box_mask = crop_box_mask[:, None, ...]
        if crop_box_mask.shape[-2:] != global_image.shape[-2:]:
            crop_box_mask = F.interpolate(crop_box_mask.float(), size=global_image.shape[-2:], mode="nearest")
        x = torch.cat([global_image, crop_box_mask.to(dtype=global_image.dtype, device=global_image.device)], dim=1)
        x = self.global_stem(x)
        x = self.global_layer1(self.global_maxpool(x))
        x = self.global_layer2(x)
        x = self.global_layer3(x)
        return self.global_layer4(x)

    def decode(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        x0, x1, x2, x3, z = features
        y = self._match_size(self.up4(z), x3)
        y = self.dec4(torch.cat([y, x3], dim=1))
        y = self._match_size(self.up3(y), x2)
        y = self.dec3(torch.cat([y, x2], dim=1))
        y = self._match_size(self.up2(y), x1)
        y = self.dec2(torch.cat([y, x1], dim=1))
        y = self._match_size(self.up1(y), x0)
        y = self.dec1(torch.cat([y, x0], dim=1))
        y = self.up0(y)
        logits = self.out(self.dec0(y))
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        return logits

    def forward(
        self,
        images: torch.Tensor,
        *,
        global_image: torch.Tensor | None = None,
        crop_box_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        original_size = images.shape[-2:]
        if global_image is None:
            global_image = images
        if crop_box_mask is None:
            crop_box_mask = torch.ones(
                (images.shape[0], 1, *images.shape[-2:]),
                dtype=images.dtype,
                device=images.device,
            )
        local_features = list(self.encode(images))
        global_z = self.encode_global(global_image, crop_box_mask)
        context = self.context_proj(self.context_pool(global_z))[:, :, None, None]
        local_features[-1] = local_features[-1] + context
        features = tuple(local_features)
        logits = self.decode(features, original_size)
        det_features = self.det_mlp(self.det_pool(features[-1]))
        return {
            "mask_logits": logits,
            "objectness_logits": self.objectness(det_features),
            "bbox": torch.sigmoid(self.bbox(det_features)),
        }


def build_segmentation_model(
    model_type: str = ROI_SEGMENTER_MODEL_TYPE,
    *,
    pretrained: bool = False,
) -> FunctionalSurfaceUNetResNet18Det1Context2B:
    if model_type != ROI_SEGMENTER_MODEL_TYPE:
        raise ValueError(f"Unsupported segmentation model_type {model_type!r}.")
    return FunctionalSurfaceUNetResNet18Det1Context2B(pretrained=pretrained)


__all__ = [
    "ROI_SEGMENTER_MODEL_TYPE",
    "FunctionalSurfaceUNetResNet18Det1Context2B",
    "build_segmentation_model",
]
