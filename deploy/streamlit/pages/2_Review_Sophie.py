"""Review Sophie - poste de controle qualite display-only."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from iqa_client import API_URL, get, post

st.set_page_config(page_title="IQA - Poste Sophie", layout="wide")
st.title("Poste Sophie - controle qualite")
st.caption(f"iqa-api: {API_URL}")
st.info("Feedback Sophie display-only : l'oracle GT reste souverain pour fermer le feedback et entrainer.")

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(os.environ.get("IQA_REPO_ROOT", DEFAULT_REPO_ROOT)).resolve()
DEFAULT_RUN_DIR = os.environ.get("IQA_SOPHIE_REPLAY_RUN_DIR", "")


def _load_events(run_dir: str) -> list[dict[str, Any]]:
    if not run_dir:
        return []
    path = _resolve_path(run_dir) / "events.jsonl"
    if not path.exists():
        st.warning(f"events.jsonl introuvable : {path}")
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    mapped = REPO_ROOT / text
    return mapped


def _decision_label(value: str | None) -> str:
    normalized = str(value or "").lower()
    if normalized in {"green", "vert", "conforme"}:
        return "Conforme predit"
    if normalized in {"red", "rouge", "defective"}:
        return "Defaut predit"
    if normalized in {"orange", "warning"}:
        return "A revoir"
    return value or "-"


def _feedback_jsonl() -> str:
    rows = st.session_state.get("sophie_feedback", [])
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else "")


if "sophie_index" not in st.session_state:
    st.session_state["sophie_index"] = 0
if "sophie_feedback" not in st.session_state:
    st.session_state["sophie_feedback"] = []

tab_replay, tab_api = st.tabs(["File replay", "Historique API"])

with tab_replay:
    run_dir = st.text_input(
        "Dossier replay lifecycle",
        value=DEFAULT_RUN_DIR,
        placeholder=".cache/iqa/replay_lifecycle/production_replay_natural/<run_id>",
    )
    events = _load_events(run_dir)
    if not events:
        st.info("Renseigne un dossier de run replay lifecycle contenant events.jsonl.")
    else:
        st.session_state["sophie_index"] = min(st.session_state["sophie_index"], len(events) - 1)
        current = events[st.session_state["sophie_index"]]

        nav_a, nav_b, nav_c, nav_d = st.columns([1, 1, 2, 2])
        if nav_a.button("Precedente", disabled=st.session_state["sophie_index"] <= 0):
            st.session_state["sophie_index"] -= 1
            st.rerun()
        if nav_b.button("Suivante", disabled=st.session_state["sophie_index"] >= len(events) - 1):
            st.session_state["sophie_index"] += 1
            st.rerun()
        nav_c.metric("Piece", f"{st.session_state['sophie_index'] + 1}/{len(events)}")
        nav_d.metric("Decision", _decision_label(current.get("decision")))

        raw_col, heatmap_col, info_col = st.columns([1.2, 1.2, 0.9])
        with raw_col:
            st.subheader("Image brute")
            image_path = _resolve_path(current.get("image_path"))
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.code(current.get("image_path") or current.get("image_uri") or "image indisponible")

        with heatmap_col:
            st.subheader("Overlay heatmap")
            heatmap_path = _resolve_path(current.get("heatmap_path"))
            if heatmap_path.exists():
                st.image(str(heatmap_path), use_container_width=True)
            elif current.get("heatmap_uri"):
                st.code(current["heatmap_uri"])
            else:
                st.warning("Aucune heatmap disponible pour cette piece.")

        with info_col:
            st.subheader("Contexte")
            st.write(f"**piece_event_id**: `{current.get('piece_event_id')}`")
            st.write(f"**lot_id**: `{current.get('lot_id')}`")
            st.write(f"**scenario**: `{current.get('scenario_id')}`")
            st.write(f"**source_class**: `{current.get('source_class')}`")
            st.write(f"**score**: `{current.get('score')}`")
            st.write(f"**seuil orange**: `{current.get('threshold_orange')}`")
            st.write(f"**seuil rouge**: `{current.get('threshold_red')}`")
            st.write(f"**ROI**: `{current.get('roi_quality_status')}` ratio `{current.get('roi_ratio')}`")
            st.write(f"**oracle GT**: `{current.get('oracle_verdict')}`")
            if current.get("roi_mask_uri"):
                st.caption(f"ROI mask: {current['roi_mask_uri']}")
            if current.get("heatmap_uri"):
                st.caption(f"Heatmap: {current['heatmap_uri']}")

        st.divider()
        st.subheader("Feedback Sophie")
        fb_col, comment_col = st.columns([1, 2])
        feedback_status = fb_col.selectbox(
            "Avis Sophie",
            options=[
                ("conforme_valide", "Conforme"),
                ("defaut_confirme", "Defaut suspecte"),
                ("roi_warning", "A revoir / ROI"),
            ],
            format_func=lambda item: item[1],
        )[0]
        comment = comment_col.text_input("Commentaire", value="")
        if st.button("Marquer revue"):
            st.session_state["sophie_feedback"].append(
                {
                    "feedback_source": "human_sophie",
                    "feedback_status": feedback_status,
                    "piece_event_id": current.get("piece_event_id"),
                    "scenario_id": current.get("scenario_id"),
                    "lot_id": current.get("lot_id"),
                    "decision": current.get("decision"),
                    "comment": comment,
                    "train_eligible": False,
                    "reason": "human_sophie_display_only",
                }
            )
            st.success("Feedback Sophie enregistre en session (display-only).")

        st.download_button(
            "Exporter feedback Sophie JSONL",
            data=_feedback_jsonl(),
            file_name="sophie_feedback.jsonl",
            mime="application/jsonl",
            disabled=not st.session_state.get("sophie_feedback"),
        )

with tab_api:
    if st.button("Rafraichir historique API"):
        st.rerun()
    try:
        rows = get("/predictions")
    except requests.RequestException as exc:
        st.error(f"iqa-api indisponible : {exc}")
        rows = []

    if not rows:
        st.info("Aucune prediction API a revoir.")
    else:
        st.dataframe(
            [
                {
                    "prediction_id": row.get("prediction_id"),
                    "lot": row.get("lot_id") or row.get("scenario_id"),
                    "piece": row.get("piece_event_id"),
                    "decision": row.get("decision"),
                    "oracle": row.get("oracle_verdict") or "-",
                    "heatmap_uri": row.get("heatmap_uri") or "-",
                    "feedback_ferme": row.get("feedback_closed"),
                    "sophie": row.get("display_feedback_status") or "-",
                }
                for row in rows
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Envoyer feedback Sophie display-only via API")
        prediction_ids = [row.get("prediction_id") for row in rows if row.get("prediction_id")]
        selected = st.selectbox("prediction_id", options=prediction_ids)
        selected_row = next(row for row in rows if row.get("prediction_id") == selected)
        api_status = st.selectbox("Avis Sophie API", options=["conforme_valide", "defaut_confirme", "roi_warning"])
        api_comment = st.text_input("Commentaire API", value="")
        if st.button("Envoyer a /feedback"):
            try:
                result = post(
                    "/feedback",
                    {
                        "prediction_id": selected,
                        "piece_event_id": selected_row.get("piece_event_id"),
                        "scenario_id": selected_row.get("scenario_id"),
                        "feedback_source": "human_sophie",
                        "feedback_status": api_status,
                        "comment": api_comment,
                    },
                )
                st.json(result)
            except requests.RequestException as exc:
                st.error(f"Feedback Sophie indisponible : {exc}")
