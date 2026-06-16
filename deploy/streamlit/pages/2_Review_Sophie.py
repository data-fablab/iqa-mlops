"""Review Sophie - revue en lecture seule (divergence oracle).

Sophie (PRD stories 1-2) veut voir les decisions Vert/Orange/Rouge et concentrer
sa revue. En MVP, `human_sophie` n'est pas operationnel : l'oracle GT est
souverain. Cette vue est donc en **lecture seule** et met en evidence les
divergences entre la decision du modele et le verdict oracle :

- faux_negatif : modele Vert mais piece defective (echappement)
- faux_positif : modele Rouge mais piece conforme (faux rejet)
- orange_a_revoir : decision Orange (a arbitrer)
"""

from __future__ import annotations

import requests
import streamlit as st
from iqa_client import API_URL, get

st.set_page_config(page_title="IQA - Review Sophie", layout="wide")
st.title("Review Sophie - revue lecture seule")
st.caption(f"iqa-api: {API_URL}")
st.info("Vue en lecture seule (MVP). L'oracle GT reste souverain ; aucun verdict humain operationnel.")

if st.button("Rafraichir"):
    st.rerun()

try:
    rows = get("/predictions")
except requests.RequestException as exc:
    st.error(f"iqa-api indisponible : {exc}")
    st.stop()

if not rows:
    st.info("Aucune prediction a revoir. Genere des predictions depuis l'Accueil.")
    st.stop()

DIVERGENCE_LABELS = {
    "faux_negatif": "Faux negatif (echappement)",
    "faux_positif": "Faux positif (faux rejet)",
    "orange_a_revoir": "Orange a revoir",
    "concordant": "Concordant",
    None: "En attente feedback",
}

lots = sorted({row["scenario_id"] for row in rows if row["scenario_id"]})
col_a, col_b = st.columns(2)
lot_filter = col_a.selectbox("Lot (scenario_id)", options=["(tous)", *lots])
only_divergent = col_b.checkbox("Uniquement les divergences", value=False)

filtered = []
for row in rows:
    if lot_filter != "(tous)" and row["scenario_id"] != lot_filter:
        continue
    if only_divergent and row["divergence"] not in {"faux_negatif", "faux_positif"}:
        continue
    filtered.append(
        {
            "prediction_id": row["prediction_id"],
            "lot": row["scenario_id"],
            "piece": row["piece_event_id"],
            "decision": row["decision"],
            "oracle": row["oracle_verdict"] or "-",
            "divergence": DIVERGENCE_LABELS.get(row["divergence"], row["divergence"]),
            "feedback_ferme": row["feedback_closed"],
            "modele": row["model_version"],
            "cree_le": row["created_at"],
        }
    )

divergences = sum(1 for row in filtered if row["divergence"].startswith("Faux"))
st.metric("Lignes affichees", len(filtered), f"{divergences} divergence(s)")

st.dataframe(filtered, use_container_width=True, hide_index=True)
