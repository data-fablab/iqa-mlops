"""Sophie vitrine MVP: lots, modele actif, statut piece, lien feedback.

Talks to the `iqa-api` FastAPI gateway. This is a Phase 1 placeholder UI:
it wires the contracts (/model/version, /replay-scenarios, /predict,
/feedback) without any PostgreSQL-backed history yet.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("IQA_API_URL", "http://localhost:8000")

st.set_page_config(page_title="IQA - Vitrine Sophie", layout="wide")
st.title("Industrial Quality Assistant - Vitrine Sophie")
st.caption(f"iqa-api: {API_URL}")

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None


def _get(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=5)
    response.raise_for_status()
    return response.json()


def _post(path: str, json: dict, headers: dict | None = None):
    response = requests.post(f"{API_URL}{path}", json=json, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()


col_model, col_lots = st.columns(2)

with col_model:
    st.header("Modele actif")
    try:
        st.json(_get("/model/version"))
    except requests.RequestException as exc:
        st.error(f"iqa-api indisponible : {exc}")

with col_lots:
    st.header("Lots (scenarios de replay)")
    try:
        st.dataframe(_get("/replay-scenarios"))
    except requests.RequestException as exc:
        st.error(f"iqa-api indisponible : {exc}")

st.divider()

st.header("Statut piece")
with st.form("predict_form"):
    piece_event_id = st.text_input("piece_event_id", value="demo-piece-0001")
    scenario_id = st.text_input("scenario_id", value="production_replay_natural")
    image_uri = st.text_input("image_uri", value="s3://iqa-ingested-images/demo-piece-0001.png")
    submitted = st.form_submit_button("Predire")

if submitted:
    try:
        result = _post(
            f"/piece-events/{piece_event_id}/predict",
            {"scenario_id": scenario_id, "image_uri": image_uri},
        )
        st.session_state["last_prediction"] = result
        st.json(result)
    except requests.RequestException as exc:
        st.error(f"Predict indisponible : {exc}")

st.divider()

st.header("Feedback (oracle GT)")
st.caption("MVP : oracle_gt ferme le feedback ; human_sophie reste limitee a l'affichage.")
last_prediction = st.session_state.get("last_prediction") or {}
last_prediction_payload = last_prediction.get("prediction", {})
last_prediction_id = last_prediction_payload.get("prediction_id", "")
if last_prediction_id:
    st.info(f"Derniere prediction disponible : {last_prediction_id}")
with st.form("feedback_form"):
    prediction_id = st.text_input("prediction_id", value=last_prediction_id)
    fb_piece_event_id = st.text_input("piece_event_id ", value="demo-piece-0001")
    fb_scenario_id = st.text_input("scenario_id ", value="production_replay_natural")
    gt_mask_has_defect = st.checkbox("gt_mask_has_defect")
    gt_mask_uri = st.text_input("gt_mask_uri (optionnel)", value="")
    fb_submitted = st.form_submit_button("Envoyer feedback")

if fb_submitted:
    service_token = os.environ.get("IQA_SERVICE_TOKEN")
    headers = {"X-IQA-Service-Token": service_token} if service_token else None
    try:
        result = _post(
            "/feedback",
            {
                "prediction_id": prediction_id,
                "piece_event_id": fb_piece_event_id,
                "scenario_id": fb_scenario_id,
                "feedback_source": "oracle_gt",
                "gt_mask_uri": gt_mask_uri or None,
                "gt_mask_has_defect": gt_mask_has_defect,
            },
            headers=headers,
        )
        st.json(result)
    except requests.RequestException as exc:
        st.error(f"Feedback indisponible : {exc}")
