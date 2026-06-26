"""PatchCore domain-drift detector (Issues 11, 19–21).

A distance-to-nominal detector that separates *product domains* (Casting_class1 vs
class2/class3) where the Feature-AE reconstruction score cannot: empirically the AE
detects **defects**, not a **domain change** (ADR 0010, amendment 2026-06-26). This
detector is the foundation of the real drift path — a signal that genuinely moves
when the product domain shifts.

It mirrors how the AE is treated: a pure seam (no serving I/O) that builds a coreset
**memory bank** of nominal patches, scores an image by its kNN distance to that bank,
and calibrates an out-of-domain threshold; plus on-disk persistence
(``memory_bank.pt`` / ``calibration.yaml`` / ``model_manifest.json``) so it reloads
without rebuilding the bank.

Canonical parameters (from the validated prototype):

- backbone ``wide_resnet50_2`` (ImageNet), features hooked on ``layer2`` + ``layer3``,
  l3 upsampled to the l2 grid, concatenated, then ``avg_pool2d(k=3,s=1,p=1)`` for a
  locally-aware patch embedding;
- image score = ``max`` over patches of ``min_j || f_patch - bank_j ||_2`` (max-patch,
  the canonical PatchCore image score);
- per-piece threshold = p90 of the scores on the union of covered hold-outs.

Multi-class coverage (Issue 19–21): the bank can cover N classes with balanced
sampling; the manifest records ``covered_classes`` as the cumulative source of truth.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# --------------------------------------------------------------------------- #
# Canonical detector parameters (validated prototype)
# --------------------------------------------------------------------------- #
DEFAULT_BACKBONE = "wide_resnet50_2"
DEFAULT_LAYERS: tuple[str, ...] = ("layer2", "layer3")
DEFAULT_MEM_IMAGES = 200
DEFAULT_CORESET_PATCHES = 25_000
DEFAULT_SEED = 42
DEFAULT_PERCENTILE = 90.0

# Drift only needs the image-level score, so the serving transform is the plain
# ImageNet Resize(256) + CenterCrop(224) (no GT mask to align — unlike the AE path).
IMAGE_RESIZE = 256
IMAGE_CROP = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IN_DOMAIN = "in_domain"
OUT_OF_DOMAIN = "out_of_domain"

MODEL_VERSION = "patchcore_domain_drift_v001"
MODEL_TYPE = "patchcore_distance_to_nominal_wrn50"
BANK_FILENAME = "memory_bank.pt"
CALIBRATION_FILENAME = "calibration.yaml"
MANIFEST_FILENAME = "model_manifest.json"

# Resident location of the registered detector (mounted RO in the GPU container).
DEFAULT_DETECTOR_DIR = "/opt/iqa/models/patchcore_domain_drift_v001"


# --------------------------------------------------------------------------- #
# Pure scoring math (unit-tested without a GPU / backbone)
# --------------------------------------------------------------------------- #
def max_patch_score(patch_features: torch.Tensor, bank: torch.Tensor) -> float:
    """Canonical PatchCore image score: ``max`` over patches of NN distance to the bank.

    ``patch_features`` is ``(P, C)`` (one image's locally-aware patches), ``bank`` is
    ``(M, C)`` nominal patches. Higher = more out-of-domain.
    """
    if patch_features.ndim != 2 or bank.ndim != 2:
        raise ValueError(
            f"expected 2D (P,C)/(M,C) tensors, got {tuple(patch_features.shape)} / {tuple(bank.shape)}"
        )
    distances = torch.cdist(patch_features, bank)  # (P, M)
    nearest = distances.min(dim=1).values  # per-patch NN distance
    return float(nearest.max().item())


def coreset_subsample(patches: torch.Tensor, n_patches: int, seed: int = DEFAULT_SEED) -> torch.Tensor:
    """Deterministically subsample ``n_patches`` rows from ``(N, C)`` (seeded)."""
    if patches.ndim != 2:
        raise ValueError(f"expected a 2D (N,C) tensor, got {tuple(patches.shape)}")
    if patches.shape[0] <= n_patches:
        return patches
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    index = torch.randperm(patches.shape[0], generator=generator)[:n_patches]
    return patches[index]


def balanced_pool(
    images_by_class: dict[str, list[str | Path]],
    budget: int,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Draw ``budget`` images balanced equally across classes (seeded, deterministic)."""
    rng = random.Random(seed)
    classes = sorted(images_by_class)
    if not classes:
        return []
    per_class = budget // len(classes)
    remainder = budget % len(classes)
    pool: list[str] = []
    for idx, cls in enumerate(classes):
        paths = [str(p) for p in images_by_class[cls]]
        shuffled = list(paths)
        rng.shuffle(shuffled)
        take = per_class + (1 if idx < remainder else 0)
        pool.extend(shuffled[:take])
    return pool


def calibrate_threshold(scores: Iterable[float], percentile: float = DEFAULT_PERCENTILE) -> float:
    """Per-piece out-of-domain threshold = ``percentile`` of nominal hold-out scores."""
    values = np.asarray([float(s) for s in scores], dtype=np.float64)
    if values.size == 0:
        raise ValueError("calibration needs at least one score")
    return float(np.percentile(values, float(percentile)))


def regime_for_score(score: float, threshold: float) -> str:
    """Map a drift score to a regime via the calibrated per-piece threshold."""
    return OUT_OF_DOMAIN if float(score) >= float(threshold) else IN_DOMAIN


@dataclass(frozen=True)
class DomainDriftCalibration:
    """The calibrated per-piece threshold plus provenance stats."""

    threshold: float
    percentile: float = DEFAULT_PERCENTILE
    class1_score_median: float | None = None
    class1_sample_count: int = 0
    holdout_sample_count: int = 0

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "threshold": float(self.threshold),
            "percentile": float(self.percentile),
            "class1_score_median": (
                None if self.class1_score_median is None else float(self.class1_score_median)
            ),
            "class1_sample_count": int(self.class1_sample_count),
            "holdout_sample_count": int(self.holdout_sample_count),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DomainDriftCalibration":
        return cls(
            threshold=float(payload["threshold"]),
            percentile=float(payload.get("percentile", DEFAULT_PERCENTILE)),
            class1_score_median=(
                None
                if payload.get("class1_score_median") is None
                else float(payload["class1_score_median"])
            ),
            class1_sample_count=int(payload.get("class1_sample_count", 0)),
            holdout_sample_count=int(payload.get("holdout_sample_count", 0)),
        )


# --------------------------------------------------------------------------- #
# Detector (backbone-dependent; the seam composes the pure functions above)
# --------------------------------------------------------------------------- #
class PatchCoreDomainDriftDetector:
    """Distance-to-nominal domain-drift detector with a coreset memory bank."""

    DEFAULT_COVERED_CLASSES: list[str] = ["Casting_class1"]

    def __init__(
        self,
        *,
        bank: torch.Tensor | None = None,
        calibration: DomainDriftCalibration | None = None,
        device: str | None = None,
        seed: int = DEFAULT_SEED,
        layers: Sequence[str] = DEFAULT_LAYERS,
        backbone_name: str = DEFAULT_BACKBONE,
        coreset_patches: int = DEFAULT_CORESET_PATCHES,
        covered_classes: Sequence[str] | None = None,
    ) -> None:
        requested = device or "cuda"
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self.device = requested
        self.seed = int(seed)
        self.layers = tuple(layers)
        self.backbone_name = backbone_name
        self.coreset_patches = int(coreset_patches)
        self.covered_classes: list[str] = sorted(set(covered_classes)) if covered_classes else list(self.DEFAULT_COVERED_CLASSES)
        self.calibration = calibration
        self._bank = None if bank is None else bank.to(self.device)
        self._backbone: torch.nn.Module | None = None
        self._features: dict[str, torch.Tensor] = {}
        self._transform = None

    # ---- backbone (lazy; needs torchvision + ImageNet weights) ----
    def _ensure_backbone(self) -> None:
        if self._backbone is not None:
            return
        import torchvision
        from torchvision import transforms

        weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = torchvision.models.wide_resnet50_2(weights=weights).to(self.device).eval()
        self._features = {}
        for layer_name in self.layers:
            module = getattr(backbone, layer_name)
            module.register_forward_hook(
                lambda _m, _i, output, key=layer_name: self._features.__setitem__(key, output)
            )
        self._backbone = backbone
        self._transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_RESIZE),
                transforms.CenterCrop(IMAGE_CROP),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
            ]
        )

    @torch.no_grad()
    def patch_embeds(self, image_paths: Sequence[str | Path], batch: int = 8) -> torch.Tensor:
        """Locally-aware patch features ``(n_images, n_patches, C)`` on the l2 grid."""
        from PIL import Image

        self._ensure_backbone()
        assert self._backbone is not None and self._transform is not None
        out: list[torch.Tensor] = []
        paths = [str(p) for p in image_paths]
        for start in range(0, len(paths), batch):
            chunk = paths[start : start + batch]
            images = torch.stack(
                [self._transform(Image.open(p).convert("RGB")) for p in chunk]
            ).to(self.device)
            self._features.clear()
            self._backbone(images)
            fused = self._fuse_feature_maps()
            batch_n, channels, height, width = fused.shape
            fused = fused.permute(0, 2, 3, 1).reshape(batch_n, height * width, channels)
            out.append(fused.cpu())
        return torch.cat(out, dim=0)

    def _fuse_feature_maps(self) -> torch.Tensor:
        """Concatenate the hooked layers on the first layer's grid, then local-avg-pool."""
        primary = self._features[self.layers[0]]
        maps = [primary]
        for layer_name in self.layers[1:]:
            upsampled = F.interpolate(
                self._features[layer_name], size=primary.shape[-2:], mode="bilinear", align_corners=False
            )
            maps.append(upsampled)
        fused = torch.cat(maps, dim=1)
        return F.avg_pool2d(fused, kernel_size=3, stride=1, padding=1)

    # ---- bank construction / scoring / calibration ----
    @property
    def bank(self) -> torch.Tensor | None:
        return self._bank

    def build_bank(self, image_paths: Sequence[str | Path]) -> torch.Tensor:
        """Build the coreset memory bank from nominal (class1/good) images.

        The caller is responsible for excluding the calibration hold-out from
        ``image_paths`` (the bank and the hold-out must be disjoint).
        """
        embeds = self.patch_embeds(image_paths)  # (n, P, C)
        flat = embeds.reshape(-1, embeds.shape[-1])  # (N, C)
        coreset = coreset_subsample(flat, self.coreset_patches, seed=self.seed)
        self._bank = coreset.to(self.device)
        return self._bank

    def score(self, image_path: str | Path) -> float:
        if self._bank is None:
            raise RuntimeError("memory bank is not built/loaded; call build_bank or load first")
        embeds = self.patch_embeds([image_path]).to(self.device)  # (1, P, C)
        return max_patch_score(embeds[0], self._bank)

    def calibrate(
        self,
        holdout_paths: Sequence[str | Path],
        *,
        percentile: float = DEFAULT_PERCENTILE,
    ) -> DomainDriftCalibration:
        """Calibrate the per-piece threshold = ``percentile`` of the union hold-out scores."""
        scores = [self.score(path) for path in holdout_paths]
        threshold = calibrate_threshold(scores, percentile)
        self.calibration = DomainDriftCalibration(
            threshold=threshold,
            percentile=percentile,
            class1_score_median=float(np.median(scores)) if scores else None,
            class1_sample_count=len(scores),
            holdout_sample_count=len(scores),
        )
        return self.calibration

    def regime(self, score: float) -> str:
        if self.calibration is None:
            raise RuntimeError("detector is not calibrated; call calibrate or load first")
        return regime_for_score(score, self.calibration.threshold)

    # ---- persistence ----
    def save(self, directory: str | Path) -> Path:
        """Persist bank + calibration + manifest so the detector reloads without a rebuild."""
        if self._bank is None or self.calibration is None:
            raise RuntimeError("cannot save an unbuilt/uncalibrated detector")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self._bank.cpu(), target / BANK_FILENAME)
        (target / CALIBRATION_FILENAME).write_text(
            yaml.safe_dump({"domain_drift_calibration": self.calibration.to_dict()}, sort_keys=False),
            encoding="utf-8",
        )
        (target / MANIFEST_FILENAME).write_text(
            json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    def manifest(self) -> dict:
        """Repo-convention manifest describing the registered detector."""
        bank_shape = list(self._bank.shape) if self._bank is not None else None
        calibration = self.calibration.to_dict() if self.calibration is not None else None
        return {
            "model_version": MODEL_VERSION,
            "model_type": MODEL_TYPE,
            "artifact_uri": f"s3://iqa-models/{MODEL_VERSION}/{BANK_FILENAME}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backbone": self.backbone_name,
            "feature_layers": list(self.layers),
            "coreset_patches": self.coreset_patches,
            "seed": self.seed,
            "memory_bank_shape": bank_shape,
            "image_resize": IMAGE_RESIZE,
            "image_crop": IMAGE_CROP,
            "score_image": "max_patch_knn_l2",
            "regime_threshold": calibration["threshold"] if calibration else None,
            "calibration": calibration,
            "covered_classes": list(self.covered_classes),
            "signal": "domain_drift",
            "purpose": "domain_drift_only_not_defect_detection",
        }

    @classmethod
    def load(
        cls, directory: str | Path, *, device: str | None = None, **kwargs
    ) -> "PatchCoreDomainDriftDetector":
        """Reload a registered detector (bank + calibration) without rebuilding."""
        source = Path(directory)
        bank = torch.load(source / BANK_FILENAME, map_location="cpu")
        calibration_doc = yaml.safe_load((source / CALIBRATION_FILENAME).read_text(encoding="utf-8"))
        calibration = DomainDriftCalibration.from_dict(calibration_doc["domain_drift_calibration"])
        manifest_path = source / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        return cls(
            bank=bank,
            calibration=calibration,
            device=device,
            seed=int(manifest.get("seed", DEFAULT_SEED)),
            layers=tuple(manifest.get("feature_layers", DEFAULT_LAYERS)),
            backbone_name=manifest.get("backbone", DEFAULT_BACKBONE),
            coreset_patches=int(manifest.get("coreset_patches", DEFAULT_CORESET_PATCHES)),
            covered_classes=manifest.get("covered_classes"),
            **kwargs,
        )


def union_covered_classes(current: Sequence[str], triggering_class: str) -> list[str]:
    """Return the sorted, deduplicated union of ``current`` covered classes and ``triggering_class``."""
    return sorted(set(current) | {triggering_class})


__all__ = [
    "DEFAULT_DETECTOR_DIR",
    "IN_DOMAIN",
    "OUT_OF_DOMAIN",
    "MODEL_VERSION",
    "DomainDriftCalibration",
    "PatchCoreDomainDriftDetector",
    "balanced_pool",
    "calibrate_threshold",
    "coreset_subsample",
    "max_patch_score",
    "regime_for_score",
    "union_covered_classes",
]
