"""Dashboard Marc - supervision par lot.

Marc (PRD story 3) veut suivre par lot les volumes, le taux Orange, les Rouges.
Lecture des KPIs agreges via l'API (`/lots/summary`).
"""

from __future__ import annotations

import requests
import streamlit as st
from iqa_client import API_URL, get

st.set_page_config(page_title="IQA - Dashboard Marc", layout="wide")
st.title("Dashboard Marc - supervision par lot")
st.caption(f"iqa-api: {API_URL}")

if st.button("Rafraichir"):
    st.rerun()

try:
    lots = get("/lots/summary")
except requests.RequestException as exc:
    st.error(f"iqa-api indisponible : {exc}")
    st.stop()

if not lots:
    st.info("Aucune prediction enregistree pour le moment. Genere des predictions depuis l'Accueil.")
    st.stop()

total = sum(lot["total"] for lot in lots)
rouges = sum(lot["rouge"] for lot in lots)
oranges = sum(lot["orange"] for lot in lots)
divergences = sum(lot["divergences"] for lot in lots)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Volume total", total)
k2.metric("Oranges", oranges, f"{round(100 * oranges / total, 1)} %")
k3.metric("Rouges", rouges, f"{round(100 * rouges / total, 1)} %")
k4.metric("Divergences oracle", divergences)

st.divider()
st.subheader("Par lot (scenario_id)")
st.dataframe(
    lots,
    use_container_width=True,
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

st.subheader("Distribution Vert / Orange / Rouge par lot")
chart_data = {
    "Vert": {lot["scenario_id"]: lot["vert"] for lot in lots},
    "Orange": {lot["scenario_id"]: lot["orange"] for lot in lots},
    "Rouge": {lot["scenario_id"]: lot["rouge"] for lot in lots},
}
st.bar_chart(chart_data, color=["#2ca02c", "#ff7f0e", "#d62728"])
