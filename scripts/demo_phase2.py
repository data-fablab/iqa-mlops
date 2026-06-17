"""Script de demo Phase 2 IQA.

Rejoue un parcours de bout en bout contre une API IQA demarree :

1. verifie la sante de l'API ;
2. genere des predictions sur plusieurs lots ;
3. ferme des feedbacks oracle GT, dont un defective qui cree une divergence
   (faux negatif : modele Vert vs oracle defective) ;
4. affiche l'agregat par lot (vue Marc) et les divergences (vue Sophie) ;
5. rappelle les URLs des vues Streamlit et du dashboard Grafana.

Sans dependance externe (urllib). Configurable par variables d'environnement :
  IQA_API_URL       (defaut http://localhost:8000)
  IQA_SERVICE_TOKEN (si l'API exige un token de service pour /feedback)
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

API_URL = os.environ.get("IQA_API_URL", "http://localhost:8000")
SERVICE_TOKEN = os.environ.get("IQA_SERVICE_TOKEN")

# (piece_event_id, scenario_id, oracle dit "defective" ?)
DEMO_PIECES = [
    ("demo-pe-001", "lot_demo_A", False),
    ("demo-pe-002", "lot_demo_A", True),  # oracle defective -> faux negatif
    ("demo-pe-003", "lot_demo_B", False),
    ("demo-pe-004", "lot_demo_B", False),
]


def _request(method: str, path: str, payload: dict | None = None) -> dict | list:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if SERVICE_TOKEN and path == "/feedback":
        headers["X-IQA-Service-Token"] = SERVICE_TOKEN
    req = urllib.request.Request(f"{API_URL}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as response:  # noqa: S310 (URL maitrisee)
        return json.loads(response.read().decode())


def _predict(piece_event_id: str, scenario_id: str) -> str:
    result = _request(
        "POST",
        f"/piece-events/{piece_event_id}/predict",
        {"scenario_id": scenario_id, "image_uri": f"s3://iqa-ingested-images/{piece_event_id}.png"},
    )
    prediction = result["prediction"]
    print(f"  predict {piece_event_id} ({scenario_id}) -> {prediction['decision']}")
    return prediction["prediction_id"]


def _feedback(prediction_id: str, piece_event_id: str, scenario_id: str, defective: bool) -> None:
    _request(
        "POST",
        "/feedback",
        {
            "prediction_id": prediction_id,
            "piece_event_id": piece_event_id,
            "scenario_id": scenario_id,
            "feedback_source": "oracle_gt",
            "gt_mask_has_defect": defective,
        },
    )
    verdict = "defective" if defective else "conforme"
    print(f"  feedback oracle {piece_event_id} -> {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-feedback", action="store_true", help="Predire sans fermer le feedback.")
    args = parser.parse_args()

    print(f"== Demo Phase 2 IQA (api: {API_URL}) ==\n")

    try:
        health = _request("GET", "/health")
    except urllib.error.URLError as exc:
        raise SystemExit(f"API injoignable sur {API_URL} : {exc}") from exc
    print(f"1. Sante API : {health}\n")

    print("2. Predictions :")
    closed = []
    for piece_event_id, scenario_id, defective in DEMO_PIECES:
        prediction_id = _predict(piece_event_id, scenario_id)
        closed.append((prediction_id, piece_event_id, scenario_id, defective))
    print()

    if not args.skip_feedback:
        print("3. Feedback oracle GT :")
        for prediction_id, piece_event_id, scenario_id, defective in closed:
            _feedback(prediction_id, piece_event_id, scenario_id, defective)
        print()

    print("4. Vue Marc - agregat par lot (/lots/summary) :")
    print(json.dumps(_request("GET", "/lots/summary"), indent=2, ensure_ascii=False))
    print()

    print("5. Vue Sophie - divergences oracle (/predictions) :")
    rows = _request("GET", "/predictions")
    divergent = [r for r in rows if r["divergence"] in {"faux_negatif", "faux_positif"}]
    for row in divergent:
        print(f"  {row['piece_event_id']} ({row['scenario_id']}): "
              f"{row['decision']} vs oracle {row['oracle_verdict']} -> {row['divergence']}")
    if not divergent:
        print("  (aucune divergence)")
    print()

    print("6. Visualiser :")
    print("   - Streamlit (Accueil / Dashboard Marc / Review Sophie) : http://localhost:8501")
    print("   - Grafana 'IQA - Vue d'ensemble'                        : http://localhost:3000")
    print("   - Prometheus targets                                    : http://localhost:9090/targets")


if __name__ == "__main__":
    main()
