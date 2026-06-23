"""Reproducible training loop for the retained RD Feature-AE."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from iqa.datasets import TiledFeatureAEDataset
from iqa.models.feature_ae import (
    REFERENCE_FEATURE_AE_CONTRACT,
    DEFAULT_FEATURE_LAYERS,
    FEATURE_AE_MODEL_TYPE,
    ReverseDistillationGatedDualContextResNet18,
    ResNetTeacherFeatures,
    feature_reconstruction_loss,
    normalize_feature_layers,
)
from iqa.roi import load_roi_mask_lookup
from iqa.training.feature_ae_evaluation import (
    FeatureAEEvaluationConfig,
    evaluate_feature_ae_checkpoint,
    update_metric_best_checkpoints,
)
from iqa.training.feature_ae_contracts import (
    CANONICAL_FEATURE_AE_PREPROCESSING,
    FEATURE_AE_BUSINESS_METRIC_PRIORITY,
    FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
    assert_canonical_feature_ae_preprocessing,
    canonical_feature_ae_preprocessing_dict,
)


REPLAY_SCENARIOS = {"production_replay_natural", "drift_domain_extension"}


@dataclass(frozen=True)
class FeatureAETrainingConfig:
    manifest_path: Path
    image_root: Path
    output_checkpoint: Path
    category: str = ""
    image_size: int = CANONICAL_FEATURE_AE_PREPROCESSING.image_size
    context_size: int = CANONICAL_FEATURE_AE_PREPROCESSING.context_size
    preprocessing_mode: str = CANONICAL_FEATURE_AE_PREPROCESSING.preprocessing_mode
    tile_stride: int = CANONICAL_FEATURE_AE_PREPROCESSING.tile_stride
    tile_train_sampling: str = CANONICAL_FEATURE_AE_PREPROCESSING.tile_train_sampling
    batch_size: int = 16
    epochs: int = 14
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    max_steps: int | None = None
    device: str = "cpu"
    pretrained_teacher: bool = True
    loss: str = "l2_cosine"
    cosine_weight: float = 0.5
    layer_loss_weights: dict[str, float] | None = None
    layers: tuple[str, ...] = DEFAULT_FEATURE_LAYERS
    repeat_factor: int = 2
    val_fraction: float = 0.15
    roi_predictions_dirs: tuple[Path, ...] = ()
    roi_threshold: float = CANONICAL_FEATURE_AE_PREPROCESSING.roi_threshold
    roi_loss_weight: float = 1.0
    background_loss_weight: float = 0.02
    min_roi_ratio: float = CANONICAL_FEATURE_AE_PREPROCESSING.min_roi_ratio
    lr_scheduler: str = "plateau"
    lr_patience: int = 4
    lr_factor: float = 0.5
    early_stopping_patience: int = 6
    metric_early_stopping_patience: int = 4
    require_business_metric_for_early_stopping: bool = False
    allow_noncanonical_preprocessing: bool = False
    checkpoint_every_epochs: int = 1
    save_best: bool = True
    scenario_id: str = ""
    dataset_version: str = ""
    manifest_version: str = ""
    candidate_version: str = ""
    roi_model_version: str = ""
    feature_ae_version: str = ""
    run_name: str = ""
    initial_checkpoint_path: Path | None = None
    initial_checkpoint_policy: str = "fresh"
    metric_eval_manifest_path: Path | None = None
    metric_eval_device: str | None = None
    metric_eval_roi_predictions_dirs: tuple[Path, ...] = ()
    gt_masks_manifest: Path | None = None
    validation_set_id: str = "validation_set_v001"
    metric_eval_every_epochs: int = 0
    metric_eval_start_epoch: int = 1
    metric_eval_batch_size: int = 8
    metric_eval_tile_stride: int | None = None
    metric_eval_layer_weights: dict[str, float] | None = None
    metric_eval_calibrate_normal: bool = False
    metric_eval_calibration_mode: str = "per_layer"
    metric_eval_calibration_stat: str = "median_mad"
    metric_eval_calibration_max_images: int = 120
    metric_eval_score_region: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_region
    metric_eval_apply_score_region_to_map: bool = False
    metric_eval_score_smoothing: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_smoothing
    metric_eval_score_image: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_image
    metric_eval_topk_fraction: float = CANONICAL_FEATURE_AE_PREPROCESSING.topk_fraction
    metric_eval_save_score_maps: bool = False
    metric_eval_save_previews: bool = False
    metric_eval_max_previews: int = 31


def train_feature_ae(config: FeatureAETrainingConfig) -> dict[str, Any]:
    _validate_config(config)
    layers = normalize_feature_layers(config.layers)
    layer_weights = config.layer_loss_weights or {}
    device = torch.device(config.device)
    roi_lookup = load_roi_mask_lookup(tuple(config.roi_predictions_dirs))
    dataset = TiledFeatureAEDataset(
        config.manifest_path,
        config.image_root,
        tile_size=config.image_size,
        context_size=config.context_size,
        tile_stride=config.tile_stride,
        repeat_factor=config.repeat_factor,
        roi_masks=roi_lookup.masks,
        roi_status=roi_lookup.status,
        roi_threshold=config.roi_threshold,
        min_roi_ratio=config.min_roi_ratio,
        train_only_normal=True,
        reject_roi_not_ok=True,
    )
    train_dataset, val_dataset = _split_dataset(dataset, config.val_fraction)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0) if val_dataset else None

    model = ReverseDistillationGatedDualContextResNet18(layers=layers).to(device)
    if config.initial_checkpoint_path is not None:
        _load_initial_checkpoint(model, config.initial_checkpoint_path, map_location=device)
    teacher = ResNetTeacherFeatures(layers=layers, pretrained=config.pretrained_teacher).to(device)
    teacher.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config.lr_patience,
            factor=config.lr_factor,
        )
        if config.lr_scheduler == "plateau"
        else None
    )

    run_dir = config.output_checkpoint.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    metric_eval_configured = config.metric_eval_every_epochs > 0 and config.metric_eval_manifest_path is not None
    best_business_score: tuple[float, ...] | None = None
    best_business_metric = ""
    best_business_metric_value: float | None = None
    metric_no_improve_epochs = 0
    metric_early_stopped = False
    epoch_metric_history: list[dict[str, Any]] = []
    step = 0
    last_checkpoint = run_dir / "checkpoint_last.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss, step = _run_epoch(
            model,
            teacher,
            train_loader,
            optimizer=optimizer,
            device=device,
            cosine_weight=config.cosine_weight,
            layer_weights=layer_weights,
            roi_loss_weight=config.roi_loss_weight,
            background_loss_weight=config.background_loss_weight,
            max_steps=config.max_steps,
            step=step,
        )
        val_loss = (
            _run_epoch(
                model,
                teacher,
                val_loader,
                optimizer=None,
                device=device,
                cosine_weight=config.cosine_weight,
                layer_weights=layer_weights,
                roi_loss_weight=config.roi_loss_weight,
                background_loss_weight=config.background_loss_weight,
                max_steps=None,
                step=step,
            )[0]
            if val_loader is not None
            else train_loss
        )
        if scheduler is not None:
            scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})

        checkpoint = _save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            config=config,
            layers=layers,
            epoch=epoch,
            step=step,
            train_samples=len(train_dataset),
            val_samples=len(val_dataset) if val_dataset else 0,
            history=history,
            best_epoch=best_epoch,
            best_loss=best_loss,
        )
        if config.output_checkpoint != last_checkpoint and not metric_eval_configured:
            shutil.copy2(last_checkpoint, config.output_checkpoint)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            if config.save_best:
                shutil.copy2(last_checkpoint, run_dir / "checkpoint_best_loss.pt")
                if not metric_eval_configured and not _has_metric_best(run_dir):
                    shutil.copy2(last_checkpoint, run_dir / "checkpoint.pt")
        else:
            no_improve_epochs += 1

        if config.checkpoint_every_epochs and epoch % config.checkpoint_every_epochs == 0:
            periodic = run_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            shutil.copy2(last_checkpoint, periodic)
            checkpoint = periodic

        if _should_run_metric_eval(config, epoch):
            eval_result = evaluate_feature_ae_checkpoint(
                FeatureAEEvaluationConfig(
                    checkpoint_path=checkpoint,
                    manifest_path=config.metric_eval_manifest_path or config.manifest_path,
                    image_root=config.image_root,
                    output_dir=run_dir / "metric_eval" / f"epoch_{epoch:03d}",
                    roi_predictions_dirs=config.metric_eval_roi_predictions_dirs or config.roi_predictions_dirs,
                    gt_masks_manifest=config.gt_masks_manifest,
                    validation_set_id=config.validation_set_id,
                    image_size=config.image_size,
                    context_size=config.context_size,
                    tile_stride=config.metric_eval_tile_stride or config.tile_stride,
                    batch_size=config.metric_eval_batch_size,
                    device=config.metric_eval_device or config.device,
                    layers=layers,
                    pretrained_teacher=config.pretrained_teacher,
                    layer_weights=config.metric_eval_layer_weights
                    or REFERENCE_FEATURE_AE_CONTRACT.normalized_layer_weights(),
                    calibrate_normal=config.metric_eval_calibrate_normal,
                    calibration_mode=config.metric_eval_calibration_mode,
                    calibration_stat=config.metric_eval_calibration_stat,
                    calibration_max_images=config.metric_eval_calibration_max_images,
                    score_region=config.metric_eval_score_region,
                    roi_threshold=config.roi_threshold,
                    apply_score_region_to_map=config.metric_eval_apply_score_region_to_map,
                    score_smoothing=config.metric_eval_score_smoothing,
                    score_image=config.metric_eval_score_image,
                    topk_fraction=config.metric_eval_topk_fraction,
                    save_score_maps=config.metric_eval_save_score_maps,
                    save_previews=config.metric_eval_save_previews,
                    max_previews=config.metric_eval_max_previews,
                )
            )
            epoch_metric_history.append(
                {
                    "epoch": epoch,
                    "checkpoint": str(checkpoint),
                    "metrics": eval_result.get("metrics") or {},
                    "per_class_metrics": eval_result.get("per_class_metrics") or {},
                    "aupimo_stability": eval_result.get("aupimo_stability") or {},
                    "predictions_path": eval_result.get("predictions_path"),
                }
            )
            _append_jsonl(run_dir / "epoch_metrics.jsonl", epoch_metric_history[-1])
            update_metric_best_checkpoints(
                run_dir=run_dir,
                candidate_checkpoint=checkpoint,
                metrics=eval_result["metrics"],
                epoch=epoch,
            )
            business_score = _business_metric_score(eval_result["metrics"])
            if business_score is None:
                if config.require_business_metric_for_early_stopping:
                    raise ValueError(
                        "Feature-AE bootstrap requires at least one business metric: "
                        + ", ".join(FEATURE_AE_BUSINESS_METRIC_PRIORITY)
                    )
            elif best_business_score is None or business_score > best_business_score:
                best_business_score = business_score
                best_business_metric, best_business_metric_value = _top_available_business_metric(eval_result["metrics"])
                metric_no_improve_epochs = 0
            else:
                metric_no_improve_epochs += 1
                if (
                    config.metric_early_stopping_patience
                    and metric_no_improve_epochs >= config.metric_early_stopping_patience
                ):
                    metric_early_stopped = True
                    break

        if config.max_steps is not None and step >= config.max_steps:
            break
        if (
            not metric_eval_configured
            and config.early_stopping_patience
            and no_improve_epochs >= config.early_stopping_patience
        ):
            break

    if config.require_business_metric_for_early_stopping and metric_eval_configured and best_business_score is None:
        raise ValueError(
            "Feature-AE bootstrap finished without any usable business metric: "
            + ", ".join(FEATURE_AE_BUSINESS_METRIC_PRIORITY)
        )

    _write_history(run_dir / "loss_history.csv", history)
    (run_dir / "metric_eval_history.json").write_text(
        json.dumps(epoch_metric_history, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "params.json").write_text(json.dumps(_metadata(config, layers), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "model_type": FEATURE_AE_MODEL_TYPE,
        "checkpoint": str(config.output_checkpoint),
        "run_dir": str(run_dir),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "steps": step,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "best_business_metric": best_business_metric,
        "best_business_metric_value": best_business_metric_value,
        "epoch_metric_history": epoch_metric_history,
        "checkpoint_selection_policy": "business_metric_only" if metric_eval_configured else "loss_only_no_metric_eval",
        "metric_early_stopped": metric_early_stopped,
        "preprocessing_mode": config.preprocessing_mode,
        "preprocessing_contract_version": FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
    }


def _business_metric_score(metrics: dict[str, Any]) -> tuple[float, ...] | None:
    score: list[float] = []
    has_metric = False
    for metric_name in FEATURE_AE_BUSINESS_METRIC_PRIORITY:
        value = metrics.get(metric_name)
        if value is None:
            score.append(float("-inf"))
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            score.append(float("-inf"))
            continue
        has_metric = True
        score.append(numeric)
    return tuple(score) if has_metric else None


def _top_available_business_metric(metrics: dict[str, Any]) -> tuple[str, float]:
    for metric_name in FEATURE_AE_BUSINESS_METRIC_PRIORITY:
        value = metrics.get(metric_name)
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            return metric_name, numeric
    raise ValueError("No finite Feature-AE business metric is available.")


def _run_epoch(
    model: ReverseDistillationGatedDualContextResNet18,
    teacher: ResNetTeacherFeatures,
    loader: DataLoader | None,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cosine_weight: float,
    layer_weights: dict[str, float],
    roi_loss_weight: float,
    background_loss_weight: float,
    max_steps: int | None,
    step: int,
) -> tuple[float, int]:
    if loader is None:
        return 0.0, step
    model.train(optimizer is not None)
    losses: list[float] = []
    for batch in loader:
        images = batch["image"].to(device)
        contexts = batch["context_image"].to(device)
        roi_mask = batch["roi_mask"].to(device)
        pixel_weight = roi_mask * float(roi_loss_weight) + (1.0 - roi_mask) * float(background_loss_weight)
        with torch.no_grad():
            teacher_features = teacher(images)
        with torch.set_grad_enabled(optimizer is not None):
            reconstructed = model(images, context_images=contexts)
            loss = feature_reconstruction_loss(
                teacher_features,
                reconstructed,
                cosine_weight=cosine_weight,
                pixel_weight=pixel_weight,
                layer_weights=layer_weights,
            )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
        losses.append(float(loss.detach().cpu()))
        if max_steps is not None and step >= max_steps:
            break
    return (sum(losses) / max(1, len(losses))), step


def _split_dataset(dataset: TiledFeatureAEDataset, val_fraction: float) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset | None]:
    if val_fraction <= 0 or len(dataset) < 2:
        return dataset, None
    val_size = max(1, int(round(len(dataset) * val_fraction)))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1337),
    )
    return train_dataset, val_dataset


def _save_checkpoint(
    path: Path,
    *,
    model: ReverseDistillationGatedDualContextResNet18,
    optimizer: torch.optim.Optimizer,
    config: FeatureAETrainingConfig,
    layers: tuple[str, ...],
    epoch: int,
    step: int,
    train_samples: int,
    val_samples: int,
    history: list[dict[str, float | int]],
    best_epoch: int,
    best_loss: float,
) -> Path:
    torch.save(
        {
            "model_type": FEATURE_AE_MODEL_TYPE,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": {
                **_metadata(config, layers),
                "epoch": epoch,
                "steps": step,
                "train_samples": train_samples,
                "val_samples": val_samples,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
            },
            "history": history,
            "layers": layers,
            "image_size": config.image_size,
            "context_size": config.context_size,
            "preprocessing_mode": config.preprocessing_mode,
            "normalization": CANONICAL_FEATURE_AE_PREPROCESSING.normalization,
            "preprocessing_contract_version": FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
            "train_manifest": str(config.manifest_path),
            "steps": step,
        },
        path,
    )
    return path


def _load_initial_checkpoint(
    model: ReverseDistillationGatedDualContextResNet18,
    checkpoint_path: Path,
    *,
    map_location: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


def _metadata(config: FeatureAETrainingConfig, layers: tuple[str, ...]) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, tuple):
            data[key] = [str(item) for item in value]
    return {
        **data,
        "model_type": FEATURE_AE_MODEL_TYPE,
        "layers": list(layers),
        "teacher_backbone": "resnet18",
        "normalization": CANONICAL_FEATURE_AE_PREPROCESSING.normalization,
        "preprocessing_contract": canonical_feature_ae_preprocessing_dict(),
        "preprocessing_contract_version": FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
    }


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    lines = ["epoch,train_loss,val_loss,lr"]
    for row in history:
        lines.append(f"{row['epoch']},{row['train_loss']},{row['val_loss']},{row['lr']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def _validate_config(config: FeatureAETrainingConfig) -> None:
    assert_canonical_feature_ae_preprocessing(
        preprocessing_mode=config.preprocessing_mode,
        image_size=config.image_size,
        context_size=config.context_size,
        tile_stride=config.tile_stride,
        tile_train_sampling=config.tile_train_sampling,
        roi_threshold=config.roi_threshold,
        min_roi_ratio=config.min_roi_ratio,
        score_region=config.metric_eval_score_region,
        score_smoothing=config.metric_eval_score_smoothing,
        score_image=config.metric_eval_score_image,
        topk_fraction=config.metric_eval_topk_fraction,
        allow_noncanonical_preprocessing=config.allow_noncanonical_preprocessing,
    )
    if config.loss != "l2_cosine":
        raise ValueError("Feature-AE reference training only supports loss='l2_cosine'.")
    if config.metric_eval_manifest_path is not None and config.metric_eval_every_epochs != 1:
        raise ValueError("Feature-AE metric evaluation must run every epoch; set metric_eval_every_epochs=1.")
    if config.initial_checkpoint_path is not None and not Path(config.initial_checkpoint_path).is_file():
        raise FileNotFoundError(f"Feature-AE initial checkpoint not found: {config.initial_checkpoint_path}")
    if config.scenario_id in REPLAY_SCENARIOS:
        missing = [
            name
            for name in ("dataset_version", "roi_model_version", "feature_ae_version")
            if not getattr(config, name)
        ]
        if missing:
            raise ValueError(f"Replay candidates require version metadata: {', '.join(missing)}.")


def _should_run_metric_eval(config: FeatureAETrainingConfig, epoch: int) -> bool:
    return (
        config.metric_eval_every_epochs > 0
        and config.metric_eval_manifest_path is not None
        and epoch >= config.metric_eval_start_epoch
        and epoch % config.metric_eval_every_epochs == 0
    )


def _has_metric_best(run_dir: Path) -> bool:
    return (run_dir / "metric_eval_best.json").exists()


__all__ = ["FeatureAETrainingConfig", "REPLAY_SCENARIOS", "train_feature_ae"]
