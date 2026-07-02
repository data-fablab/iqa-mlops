"""Data Lineage - lifecycle, gates, promotions et registry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from marc_lifecycle import classification_quality_rows, lifecycle_rows, read_json, read_jsonl
from repo_paths import default_repo_root

st.set_page_config(page_title="IQA - Data Lineage", layout="wide")
st.title("Data Lineage")
st.caption("Tracabilite technique des runs, cycles, gates, promotions et modeles.")

REPO_ROOT = default_repo_root(__file__)
DEFAULT_RUN_DIR = os.environ.get("IQA_LINEAGE_REPLAY_RUN_DIR") or os.environ.get("IQA_MARC_REPLAY_RUN_DIR", "")


def _resolve_path(value: str | os.PathLike[str] | None) -> Path:
    if not value:
        return Path()
    path = Path(value)
    if path.exists():
        return path
    text = str(value).replace("\\", "/")
    marker = "/iqa-mlops/"
    if marker in text:
        mapped = REPO_ROOT / text.split(marker, 1)[1]
        if mapped.exists():
            return mapped
    return REPO_ROOT / text


def _get(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in {None, ""}:
            return value
    return "-"


if st.button("Rafraichir"):
    st.rerun()

run_dir_value = st.text_input(
    "Dossier de run lifecycle / lineage",
    value=DEFAULT_RUN_DIR,
    placeholder=".cache/iqa/replay_lifecycle/production_replay_natural/<run_id>",
)
run_dir = _resolve_path(run_dir_value)
events_path = run_dir / "events.jsonl"
cycles_path = run_dir / "cycles.jsonl"
summary_path = run_dir / "summary.json"
progress_path = run_dir / "progress.json"
lifecycle_events_path = run_dir / "lifecycle_events.jsonl"

if not run_dir_value:
    st.info("Renseigne IQA_LINEAGE_REPLAY_RUN_DIR, IQA_MARC_REPLAY_RUN_DIR ou un dossier de run.")
elif not events_path.exists():
    st.warning(f"events.jsonl introuvable : {events_path}")
else:
    events = read_jsonl(events_path)
    cycles = read_jsonl(cycles_path)
    summary = read_json(summary_path)
    progress = read_json(progress_path)
    lifecycle_events = read_jsonl(lifecycle_events_path)
    lifecycle = lifecycle_rows(cycles)
    quality_by_model = classification_quality_rows(events, group_key="active_model_version")

    metadata = {**progress, **summary}
    scenario_id = _get(metadata, "scenario_id")
    run_id = _get(metadata, "run_id")
    status = _get(metadata, "status", "phase")

    k1, k2, k3 = st.columns(3)
    k1.metric("Events", len(events))
    k2.metric("Cycles", len(cycles))
    k3.metric("Journal lifecycle", len(lifecycle_events))

    st.dataframe(
        [
            {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "statut": status,
                "run_dir": str(run_dir),
            }
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Modeles actifs et finaux")
    st.dataframe(
        [
            {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "active_model_current": progress.get("active_model_version"),
                "active_model_final": summary.get("active_model_final"),
                "best_metric": summary.get("best_metric"),
                "best_metric_value": summary.get("best_metric_value"),
                "mlflow_run_id": summary.get("mlflow_run_id"),
                "candidate_checkpoint": summary.get("candidate_checkpoint"),
                "registry_stage": summary.get("registry_stage") or progress.get("registry_stage"),
            }
        ],
        use_container_width=True,
        hide_index=True,
    )

    chain = summary.get("promotion_chain") or progress.get("promotion_chain") or []
    if chain:
        st.write("**Chaine de promotion**")
        st.code(" -> ".join(str(item) for item in chain))

    st.divider()
    st.subheader("Cycles, gates et promotions")
    if lifecycle:
        st.dataframe(
            lifecycle,
            use_container_width=True,
            hide_index=True,
            column_config={
                "cycle_id": "Cycle",
                "actif_avant": "Actif avant",
                "modele": "Modele candidat",
                "vus": "Eval pieces",
                "defauts_vus": "Defauts vus",
                "selected_metric": "Metrique selectionnee",
                "selected_epoch": "Epoch",
                "selected_value": st.column_config.NumberColumn("Valeur selectionnee", format="%.6f"),
                "active_metric_value": st.column_config.NumberColumn("Actif", format="%.6f"),
                "candidate_metric_value": st.column_config.NumberColumn("Candidat", format="%.6f"),
                "metric_delta": st.column_config.NumberColumn("Delta", format="%.6f"),
                "localization_gate": "Gate localisation",
                "classification_gate": "Gate classification",
                "classification_progress_improved": "Classification progresse",
                "active_false_negatives": "FN actif",
                "candidate_false_negatives": "FN candidat",
                "fn_delta": "Delta FN",
                "active_image_recall": st.column_config.NumberColumn("Recall actif", format="%.3f"),
                "candidate_image_recall": st.column_config.NumberColumn("Recall candidat", format="%.3f"),
                "image_recall_delta": st.column_config.NumberColumn("Delta recall", format="%.3f"),
                "pixel_aupimo_1e-5_1e-3": st.column_config.NumberColumn("AUPIMO pixel", format="%.6f"),
                "pixel_ap": st.column_config.NumberColumn("Pixel AP", format="%.6f"),
                "image_ap": st.column_config.NumberColumn("Image AP", format="%.6f"),
                "image_recall": st.column_config.NumberColumn("Recall image", format="%.3f"),
                "gate": "Gate",
                "promotion": "Promotion",
                "stage": "Stage",
                "registry": "Registry",
                "mlflow_run_id": "MLflow",
            },
        )

        st.subheader("Train sets et artefacts")
        st.dataframe(
            [
                {
                    "cycle": row["cycle_id"],
                    "eval_pieces": row["vus"],
                    "defauts_vus": row["defauts_vus"],
                    "dataset_snapshot_id": row["dataset_snapshot_id"],
                    "calibration_set_id": row["calibration_set_id"],
                    "checkpoint": row["checkpoint"],
                    "metric_eval_best_path": row["metric_eval_best_path"],
                    "mlflow_run_id": row["mlflow_run_id"],
                }
                for row in lifecycle
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Aucun cycle modele dans cycles.jsonl pour ce run.")

    st.divider()
    st.subheader("Qualite par modele")
    if quality_by_model:
        st.dataframe(
            quality_by_model,
            use_container_width=True,
            hide_index=True,
            column_config={
                "active_model_version": "Modele actif",
                "pieces": "Pieces",
                "oracle_conforme": "Conformes controle qualite",
                "oracle_defective": "Defauts controle qualite",
                "true_good": "Bons acceptes",
                "defect_detected": "Defauts detectes",
                "false_negative": "Faux negatifs",
                "false_positive": "Faux positifs",
                "defect_recall": st.column_config.NumberColumn("Recall defauts", format="%.3f"),
                "alert_precision": st.column_config.NumberColumn("Precision alertes", format="%.3f"),
                "false_negative_rate": st.column_config.NumberColumn("Taux FN", format="%.3f"),
                "false_positive_rate": st.column_config.NumberColumn("Taux FP", format="%.3f"),
            },
        )
    else:
        st.info("Aucune metrique qualite par modele disponible.")

    if lifecycle_events:
        with st.expander("Evenements lifecycle bruts"):
            st.dataframe(lifecycle_events, use_container_width=True, hide_index=True)
