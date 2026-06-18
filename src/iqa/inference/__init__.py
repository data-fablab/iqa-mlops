"""Inference helpers for IQA runtime models.

Torch-heavy submodules are imported lazily (PEP 562) so that lightweight
consumers (e.g. the API gateway, which only needs ``iqa.inference.contracts``)
do not pull PyTorch transitively through this package's import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing/re-export only, never executed at runtime.
    from iqa.inference.feature_ae import FeatureAEPrediction as FeatureAEPrediction
    from iqa.inference.feature_ae import predict_feature_ae_image as predict_feature_ae_image
    from iqa.inference.model_loader import LoadedModel as LoadedModel
    from iqa.inference.model_loader import ProdModelLoader as ProdModelLoader
    from iqa.inference.piece import aggregate_piece_predictions as aggregate_piece_predictions
    from iqa.inference.pipeline import InferencePipelineResult as InferencePipelineResult
    from iqa.inference.pipeline import decision_from_roi_and_score as decision_from_roi_and_score
    from iqa.inference.segmentation import RoiSegmentationPrediction as RoiSegmentationPrediction
    from iqa.inference.segmentation import predict_roi_image as predict_roi_image

# Public attribute -> defining submodule. Resolved on first access only.
_LAZY_EXPORTS = {
    "FeatureAEPrediction": "iqa.inference.feature_ae",
    "predict_feature_ae_image": "iqa.inference.feature_ae",
    "LoadedModel": "iqa.inference.model_loader",
    "ProdModelLoader": "iqa.inference.model_loader",
    "aggregate_piece_predictions": "iqa.inference.piece",
    "InferencePipelineResult": "iqa.inference.pipeline",
    "decision_from_roi_and_score": "iqa.inference.pipeline",
    "RoiSegmentationPrediction": "iqa.inference.segmentation",
    "predict_roi_image": "iqa.inference.segmentation",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return [*globals(), *_LAZY_EXPORTS]
