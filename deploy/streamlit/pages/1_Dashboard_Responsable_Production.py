"""Dashboard Responsable Production - suivi operationnel des lots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from marc_lifecycle import read_jsonl
from repo_paths import default_repo_root

st.set_page_config(page_title="IQA - Responsable Production", layout="wide")

REPO_ROOT = default_repo_root(__file__)
DEFAULT_RUN_DIR = os.environ.get("IQA_MARC_REPLAY_RUN_DIR", "")

STATUS_RELEASE = "Liberer"
STATUS_HOLD = "Isoler"
STATUS_WAIT = "En attente"

QUALITY_OK = "Conforme"
QUALITY_NOK = "Non conforme"
QUALITY_PENDING = "A controler"

COLOR_GREEN = "#2ca02c"
COLOR_ORANGE = "#f5a623"
COLOR_RED = "#d62728"
COLOR_BLUE = "#2563eb"


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


def _is_quality_nonconforming(value: Any) -> bool:
    return str(value or "").lower() in {"defective", "defaut", "defectueux", "non_conforme", "non conforme"}


def _is_quality_conforming(value: Any) -> bool:
    return str(value or "").lower() in {"conforme", "conforming", "good", "ok"}


def _lot_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lots: dict[str, dict[str, Any]] = {}
    for event in events:
        lot_id = str(event.get("lot_id") or "lot_inconnu")
        lot = lots.setdefault(
            lot_id,
            {
                "lot_id": lot_id,
                "pieces_recues": 0,
                "pieces_conformes": 0,
                "pieces_non_conformes": 0,
                "pieces_a_controler": 0,
            },
        )
        lot["pieces_recues"] += 1
        verdict = event.get("oracle_verdict")
        if _is_quality_nonconforming(verdict):
            lot["pieces_non_conformes"] += 1
        elif _is_quality_conforming(verdict):
            lot["pieces_conformes"] += 1
        else:
            lot["pieces_a_controler"] += 1

    rows: list[dict[str, Any]] = []
    for lot in lots.values():
        pieces_controlees = int(lot["pieces_conformes"]) + int(lot["pieces_non_conformes"])
        lot["pieces_controlees"] = pieces_controlees
        lot["taux_conformite"] = round(100 * int(lot["pieces_conformes"]) / max(pieces_controlees, 1), 1)
        lot.update(_production_decision(lot))
        rows.append(lot)
    return sorted(rows, key=lambda row: (row["ordre"], row["lot_id"]))


def _production_decision(lot: dict[str, Any]) -> dict[str, Any]:
    if int(lot["pieces_non_conformes"]) > 0:
        return {
            "ordre": 1,
            "decision": STATUS_HOLD,
            "action": "Isoler le lot et alerter qualite",
            "commentaire": f"{lot['pieces_non_conformes']} piece(s) non conforme(s)",
        }
    if int(lot["pieces_a_controler"]) > 0:
        return {
            "ordre": 2,
            "decision": STATUS_WAIT,
            "action": "Attendre controle qualite",
            "commentaire": f"{lot['pieces_a_controler']} piece(s) a controler",
        }
    return {
        "ordre": 3,
        "decision": STATUS_RELEASE,
        "action": "Preparer expedition",
        "commentaire": "Lot conforme",
    }


def _summary(lots: list[dict[str, Any]]) -> dict[str, Any]:
    conformes = sum(int(row["pieces_conformes"]) for row in lots)
    non_conformes = sum(int(row["pieces_non_conformes"]) for row in lots)
    a_controler = sum(int(row["pieces_a_controler"]) for row in lots)
    controlees = conformes + non_conformes
    return {
        "lots_liberables": sum(1 for row in lots if row["decision"] == STATUS_RELEASE),
        "lots_isoles": sum(1 for row in lots if row["decision"] == STATUS_HOLD),
        "lots_en_attente": sum(1 for row in lots if row["decision"] == STATUS_WAIT),
        "pieces_recues": sum(int(row["pieces_recues"]) for row in lots),
        "pieces_controlees": controlees,
        "pieces_a_controler": a_controler,
        "taux_conformite": round(100 * conformes / max(controlees, 1), 1),
    }


def _card(label: str, value: str | int | float, detail: str, color: str) -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid #e5e7eb;
            border-left: 5px solid {color};
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            min-height: 106px;
            box-shadow: 0 1px 2px rgba(15,23,42,.08);
        ">
            <div style="font-size: 13px; color: #374151; font-weight: 700;">{label}</div>
            <div style="font-size: 34px; line-height: 1.25; color: {color}; font-weight: 750;">{value}</div>
            <div style="font-size: 12px; color: #6b7280;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _table_data(lots: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Lot": row["lot_id"],
                "Decision": row["decision"],
                "Action production": row["action"],
                "Pieces recues": row["pieces_recues"],
                "Pieces controlees": row["pieces_controlees"],
                "Non conformes": row["pieces_non_conformes"],
                "A controler": row["pieces_a_controler"],
                "Taux conforme": row["taux_conformite"],
                "Commentaire": row["commentaire"],
            }
            for row in lots
        ]
    )


def _styled_table(data: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row: pd.Series) -> list[str]:
        if row["Decision"] == STATUS_HOLD:
            return ["background-color: #fde2e2"] * len(row)
        if row["Decision"] == STATUS_WAIT:
            return ["background-color: #fff1d6"] * len(row)
        return ["background-color: #e5f5e5"] * len(row)

    return data.style.apply(row_style, axis=1).format({"Taux conforme": "{:.1f} %"})


def _quality_mix_chart(lots: list[dict[str, Any]]) -> alt.Chart:
    chart_rows: list[dict[str, Any]] = []
    for lot in lots:
        chart_rows.extend(
            [
                {"Lot": lot["lot_id"], "Etat": QUALITY_OK, "Pieces": lot["pieces_conformes"]},
                {"Lot": lot["lot_id"], "Etat": QUALITY_NOK, "Pieces": lot["pieces_non_conformes"]},
                {"Lot": lot["lot_id"], "Etat": QUALITY_PENDING, "Pieces": lot["pieces_a_controler"]},
            ]
        )
    return (
        alt.Chart(pd.DataFrame(chart_rows))
        .mark_bar(size=22)
        .encode(
            y=alt.Y("Lot:N", title="Lot", sort=[row["lot_id"] for row in lots]),
            x=alt.X("Pieces:Q", title="Pieces"),
            color=alt.Color(
                "Etat:N",
                title="Etat controle qualite",
                scale=alt.Scale(
                    domain=[QUALITY_OK, QUALITY_NOK, QUALITY_PENDING],
                    range=[COLOR_GREEN, COLOR_RED, COLOR_ORANGE],
                ),
            ),
            tooltip=["Lot", "Etat", "Pieces"],
        )
        .properties(height=max(260, 26 * len(lots)))
    )


def _conformity_chart(lots: list[dict[str, Any]]) -> alt.Chart:
    data = pd.DataFrame(
        [
            {
                "Lot": row["lot_id"],
                "Taux conforme": row["taux_conformite"],
                "Decision": row["decision"],
            }
            for row in lots
        ]
    )
    return (
        alt.Chart(data)
        .mark_bar(size=22)
        .encode(
            y=alt.Y("Lot:N", title="Lot", sort=[row["lot_id"] for row in lots]),
            x=alt.X("Taux conforme:Q", title="Taux conforme (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Decision:N",
                title="Decision production",
                scale=alt.Scale(
                    domain=[STATUS_RELEASE, STATUS_WAIT, STATUS_HOLD],
                    range=[COLOR_GREEN, COLOR_ORANGE, COLOR_RED],
                ),
            ),
            tooltip=["Lot", "Taux conforme", "Decision"],
        )
        .properties(height=max(260, 26 * len(lots)))
    )


def _priority_messages(lots: list[dict[str, Any]]) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    for lot in lots:
        if lot["decision"] == STATUS_HOLD:
            messages.append(("error", f"{lot['lot_id']} - isoler le lot : {lot['pieces_non_conformes']} piece(s) non conforme(s)."))
        elif lot["decision"] == STATUS_WAIT:
            messages.append(("warning", f"{lot['lot_id']} - attente controle qualite : {lot['pieces_a_controler']} piece(s) restent a controler."))
    if not messages:
        messages.append(("success", "Aucune action bloquante : tous les lots controles sont liberables."))
    return messages[:8]


st.sidebar.header("Source demo")
run_dir_value = st.sidebar.text_input(
    "Dossier de run production",
    value=DEFAULT_RUN_DIR,
    placeholder="Dossier contenant events.jsonl",
)
if st.sidebar.button("Rafraichir"):
    st.rerun()

run_dir = _resolve_path(run_dir_value)
events_path = run_dir / "events.jsonl"

st.title("Tableau De Bord Production")
st.caption(
    "Suivi des lots apres controle qualite : quels lots liberer, quels lots isoler, "
    "et quels lots attendent encore une validation."
)

if not run_dir_value:
    st.info("Renseigne un dossier de run dans la sidebar.")
elif not events_path.exists():
    st.warning(f"events.jsonl introuvable : {events_path}")
else:
    lots = _lot_rows(read_jsonl(events_path))
    summary = _summary(lots)

    st.markdown("### Situation Des Lots")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _card("Lots liberables", summary["lots_liberables"], "Prets pour expedition", COLOR_GREEN)
    with c2:
        _card("Lots a isoler", summary["lots_isoles"], "Pieces non conformes", COLOR_RED)
    with c3:
        _card("Lots en attente", summary["lots_en_attente"], "Pieces restant a controler", COLOR_ORANGE)
    with c4:
        _card("Pieces controlees", summary["pieces_controlees"], f"{summary['pieces_recues']} piece(s) recues", COLOR_BLUE)
    with c5:
        _card("Taux conforme", f"{summary['taux_conformite']} %", "Sur pieces controlees", COLOR_GREEN)

    st.divider()
    st.markdown("### Decisions Par Lot")
    st.dataframe(
        _styled_table(_table_data(lots)),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Lot": st.column_config.TextColumn("Lot", width="medium"),
            "Decision": st.column_config.TextColumn("Decision", width="small"),
            "Action production": st.column_config.TextColumn("Action production", width="medium"),
            "Pieces recues": st.column_config.NumberColumn("Pieces recues", width="small"),
            "Pieces controlees": st.column_config.NumberColumn("Pieces controlees", width="small"),
            "Non conformes": st.column_config.NumberColumn("Non conformes", width="small"),
            "A controler": st.column_config.NumberColumn("A controler", width="small"),
            "Taux conforme": st.column_config.NumberColumn("Taux conforme", format="%.1f %%"),
            "Commentaire": st.column_config.TextColumn("Commentaire", width="large"),
        },
    )

    st.divider()
    st.markdown("### Lecture Qualite")
    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.altair_chart(_conformity_chart(lots), use_container_width=True)
    with chart_b:
        st.altair_chart(_quality_mix_chart(lots), use_container_width=True)

    st.divider()
    st.markdown("### Actions Prioritaires")
    for level, message in _priority_messages(lots):
        if level == "error":
            st.error(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.success(message)
