"""Dashboard Marc - production, conformite lots et lifecycle modele."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from iqa_client import API_URL, get
from marc_lifecycle import aggregate_lots, lifecycle_rows, production_alerts, read_json, read_jsonl

st.set_page_config(page_title="IQA - Dashboard Marc", layout="wide")
st.title("Dashboard Marc - pilotage production")
st.caption(f"iqa-api: {API_URL}")

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(os.environ.get("IQA_REPO_ROOT", DEFAULT_REPO_ROOT)).resolve()
DEFAULT_RUN_DIR = os.environ.get("IQA_MARC_REPLAY_RUN_DIR", "")


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


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key) or 0) for row in rows)


if st.button("Rafraichir"):
    st.rerun()

tab_run, tab_api = st.tabs(["Run lifecycle", "Historique API"])

with tab_run:
    run_dir_value = st.text_input(
        "Dossier replay lifecycle",
        value=DEFAULT_RUN_DIR,
        placeholder=".cache/iqa/replay_lifecycle/production_replay_natural/<run_id>",
    )
    run_dir = _resolve_path(run_dir_value)
    events_path = run_dir / "events.jsonl"
    lots_path = run_dir / "lots.jsonl"
    cycles_path = run_dir / "cycles.jsonl"
    summary_path = run_dir / "summary.json"
    progress_path = run_dir / "progress.json"
    lifecycle_events_path = run_dir / "lifecycle_events.jsonl"

    st.caption("Sources attendues : events.jsonl, lots.jsonl, cycles.jsonl, summary.json, progress.json, lifecycle_events.jsonl")
    if not run_dir_value:
        st.info("Renseigne IQA_MARC_REPLAY_RUN_DIR ou un dossier de run lifecycle pour piloter les lots.")
    elif not events_path.exists():
        st.warning(f"events.jsonl introuvable : {events_path}")
    else:
        events = read_jsonl(events_path)
        cycles = read_jsonl(cycles_path)
        summary = read_json(summary_path)
        progress = read_json(progress_path)
        lifecycle_events = read_jsonl(lifecycle_events_path)
        active_model_current = str(
            progress.get("active_model_version") or summary.get("active_model_final") or ""
        )
        lots = aggregate_lots(events, active_model=active_model_current)
        lifecycle = lifecycle_rows(cycles)
        alerts = production_alerts(lots, cycles)

        total_pieces = len(events)
        conformes = _sum(lots, "conformes_gt")
        defauts = _sum(lots, "defauts_gt")
        oranges = _sum(lots, "orange")
        rouges = _sum(lots, "rouge")
        roi_fail = _sum(lots, "roi_fail_count")
        conformity_rate = round(100 * conformes / max(total_pieces, 1), 1)

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Lots traites", len(lots))
        k2.metric("Pieces inspectees", total_pieces)
        k3.metric("Conformite globale", f"{conformity_rate} %")
        k4.metric("Defauts GT", defauts)
        k5.metric("Orange / Rouge", f"{oranges} / {rouges}")
        k6.metric("ROI fail rate", f"{round(100 * roi_fail / max(total_pieces, 1), 2)} %")

        st.info(
            "Modele actif courant : "
            f"`{active_model_current or '-'}` | "
            f"Phase : `{progress.get('phase', 'complete' if summary else 'en cours')}` | "
            f"Registry stage : `{summary.get('registry_stage') or progress.get('registry_stage') or '-'}` | "
            f"Run : `{summary.get('run_id') or progress.get('run_id') or run_dir.name}`"
        )

        st.divider()
        st.subheader("Conformite des lots")
        st.dataframe(
            lots,
            width="stretch",
            hide_index=True,
            column_config={
                "lot_id": "Lot",
                "pieces": "Pieces",
                "conformes_gt": "Conformes GT",
                "defauts_gt": "Defauts GT",
                "vert": "Vert",
                "orange": "Orange",
                "rouge": "Rouge",
                "taux_conformite": st.column_config.NumberColumn("Conformite", format="%.1f%%"),
                "roi_fail_rate": st.column_config.NumberColumn("ROI fail", format="%.2f%%"),
                "statut_lot": "Statut lot",
                "model_actif": "Modele actif",
            },
        )

        chart_a, chart_b = st.columns(2)
        with chart_a:
            st.subheader("Tendance conformite par lot")
            st.line_chart({"Conformite %": {row["lot_id"]: row["taux_conformite"] for row in lots}})
        with chart_b:
            st.subheader("Defauts et decisions par lot")
            st.bar_chart(
                {
                    "Defauts GT": {row["lot_id"]: row["defauts_gt"] for row in lots},
                    "Orange": {row["lot_id"]: row["orange"] for row in lots},
                    "Rouge": {row["lot_id"]: row["rouge"] for row in lots},
                },
                color=["#d62728", "#ff7f0e", "#8b0000"],
            )

        st.subheader("Distribution Vert / Orange / Rouge")
        st.bar_chart(
            {
                "Vert": {row["lot_id"]: row["vert"] for row in lots},
                "Orange": {row["lot_id"]: row["orange"] for row in lots},
                "Rouge": {row["lot_id"]: row["rouge"] for row in lots},
            },
            color=["#2ca02c", "#ff7f0e", "#d62728"],
        )

        st.divider()
        st.subheader("Lifecycle Feature-AE")
        if lifecycle:
            st.caption(
                "Selection checkpoint : pixel_aupimo_1e-5_1e-3 -> pixel_ap -> image_ap -> image_auroc. "
                "val_loss reste secondaire."
            )
            st.dataframe(
                lifecycle,
                width="stretch",
                hide_index=True,
                column_config={
                    "cycle_id": "Cycle",
                    "actif_avant": "Actif avant",
                    "modele": "Modele",
                    "vus": "Eval pieces",
                    "defauts_vus": "Defauts vus",
                    "selected_metric": "Metrique selectionnee",
                    "selected_value": st.column_config.NumberColumn("Valeur", format="%.6f"),
                    "active_metric_value": st.column_config.NumberColumn("Actif", format="%.6f"),
                    "candidate_metric_value": st.column_config.NumberColumn("Candidat", format="%.6f"),
                    "metric_delta": st.column_config.NumberColumn("Delta", format="%.6f"),
                    "active_false_negatives": "FN actif",
                    "candidate_false_negatives": "FN candidat",
                    "activated_for_next_events": "Active ensuite",
                    "activation_scope": "Activation",
                    "pixel_aupimo_1e-5_1e-3": st.column_config.NumberColumn("AUPIMO pixel", format="%.6f"),
                    "pixel_ap": st.column_config.NumberColumn("Pixel AP", format="%.6f"),
                    "image_ap": st.column_config.NumberColumn("Image AP", format="%.6f"),
                    "image_auroc": st.column_config.NumberColumn("Image AUROC", format="%.6f"),
                    "image_recall": st.column_config.NumberColumn("Recall image", format="%.3f"),
                    "orange_rate": st.column_config.NumberColumn("Taux orange", format="%.3f"),
                    "gate": "Gate",
                    "promotion": "Promotion",
                    "stage": "Stage",
                    "registry": "Registry",
                    "mlflow_run_id": "MLflow",
                },
            )
            st.line_chart(
                {
                    "pixel_aupimo_1e-5_1e-3": {
                        row["cycle_id"]: row.get("pixel_aupimo_1e-5_1e-3")
                        for row in lifecycle
                        if row.get("pixel_aupimo_1e-5_1e-3") is not None
                    },
                    "pixel_ap": {
                        row["cycle_id"]: row.get("pixel_ap")
                        for row in lifecycle
                        if row.get("pixel_ap") is not None
                    },
                }
            )
            st.write("**Chaine de promotion**")
            chain = summary.get("promotion_chain") or progress.get("promotion_chain") or []
            st.code(" -> ".join(chain))
        else:
            st.info("Aucun cycle modele dans cycles.jsonl pour ce run.")

        if lifecycle_events:
            st.subheader("Journal lifecycle live")
            st.dataframe(
                lifecycle_events[-20:],
                width="stretch",
                hide_index=True,
            )

        st.divider()
        alert_col, lineage_col = st.columns([1, 1])
        with alert_col:
            st.subheader("Alertes production")
            if alerts:
                for alert in alerts[:12]:
                    st.warning(alert)
            else:
                st.success("Aucune alerte production sur ce run.")

        with lineage_col:
            st.subheader("Preuves lineage")
            st.caption("Git/DVC versionnent les contrats, MinIO stocke les artefacts lourds, MLflow Registry garde la source de verite modele.")
            lineage_rows = [
                {
                    "run_id": summary.get("run_id") or run_dir.name,
                    "active_model_final": summary.get("active_model_final"),
                    "best_metric": summary.get("best_metric"),
                    "best_metric_value": summary.get("best_metric_value"),
                    "mlflow_run_id": summary.get("mlflow_run_id"),
                    "candidate_checkpoint": summary.get("candidate_checkpoint"),
                }
            ]
            st.dataframe(lineage_rows, width="stretch", hide_index=True)
            if lifecycle:
                st.dataframe(
                    [
                        {
                            "cycle": row["cycle_id"],
                            "dataset_snapshot_id": row["dataset_snapshot_id"],
                            "calibration_set_id": row["calibration_set_id"],
                            "metric_eval_best_path": row["metric_eval_best_path"],
                            "mlflow_run_id": row["mlflow_run_id"],
                        }
                        for row in lifecycle
                    ],
                    width="stretch",
                    hide_index=True,
                )

with tab_api:
    st.subheader("Historique API par lot")
    try:
        api_lots = get("/lots/summary")
    except requests.RequestException as exc:
        st.error(f"iqa-api indisponible : {exc}")
        api_lots = []

    if not api_lots:
        st.info("Aucune prediction enregistree pour le moment. Genere des predictions depuis l'Accueil.")
    else:
        total = sum(lot["total"] for lot in api_lots)
        rouges = sum(lot["rouge"] for lot in api_lots)
        oranges = sum(lot["orange"] for lot in api_lots)
        divergences = sum(lot["divergences"] for lot in api_lots)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Volume total", total)
        k2.metric("Oranges", oranges, f"{round(100 * oranges / total, 1)} %")
        k3.metric("Rouges", rouges, f"{round(100 * rouges / total, 1)} %")
        k4.metric("Divergences oracle", divergences)

        st.dataframe(
            api_lots,
            width="stretch",
            column_config={
                "scenario_id": "Lot",
                "total": "Volume",
                "vert": "Vert",
                "orange": "Orange",
                "rouge": "Rouge",
                "taux_orange": st.column_config.NumberColumn("Taux Orange", format="%.1f%%"),
                "taux_rouge": st.column_config.NumberColumn("Taux Rouge", format="%.1f%%"),
                "feedback_closed": "Feedback fermes",
                "divergences": "Divergences",
            },
        )
        st.bar_chart(
            {
                "Vert": {lot["scenario_id"]: lot["vert"] for lot in api_lots},
                "Orange": {lot["scenario_id"]: lot["orange"] for lot in api_lots},
                "Rouge": {lot["scenario_id"]: lot["rouge"] for lot in api_lots},
            },
            color=["#2ca02c", "#ff7f0e", "#d62728"],
        )
