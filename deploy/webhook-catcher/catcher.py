"""Webhook catcher local pour la démo drift (décision 8).

Reçoit les notifications d'Alertmanager sur POST /alert et les imprime sur stdout
(visible à l'écran via ``docker compose logs -f webhook-catcher``). Zéro dépendance
externe : stdlib only, tourne sur python:3.12-slim. GET / expose la dernière alerte
reçue pour inspection rapide.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_LAST: dict[str, object] = {"received": None}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict[str, object]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        if self.path == "/healthz":
            self._send(200, {"status": "ok"})
            return
        self._send(200, {"last_alert": _LAST["received"]})

    def do_POST(self) -> None:  # noqa: N802 (stdlib API)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {"_raw": raw.decode("utf-8", "replace")}

        now = datetime.now(timezone.utc).isoformat()
        _LAST["received"] = {"at": now, "payload": payload}
        for alert in payload.get("alerts", []) if isinstance(payload, dict) else []:
            labels = alert.get("labels", {})
            print(
                f"[{now}] ALERTE {labels.get('alertname', '?')} "
                f"status={alert.get('status', '?')} "
                f"severity={labels.get('severity', '?')} "
                f"-> {alert.get('annotations', {}).get('summary', '')}",
                flush=True,
            )
        self._send(200, {"status": "received", "at": now})

    def log_message(self, *_args: object) -> None:  # silence default access log noise
        return


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)  # noqa: S104 (conteneur démo)
    print("webhook-catcher en écoute sur :8080 (POST /alert)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
