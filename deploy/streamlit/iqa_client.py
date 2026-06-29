"""Shared IQA API helpers for the Streamlit Sophie/Marc views."""

from __future__ import annotations

import os
from typing import Any

import requests

API_URL = os.environ.get("IQA_API_URL", "http://localhost:8000")

# Browser-facing base URL for assets the user's browser fetches directly (e.g. the
# /heatmap image proxy embedded via st.image / ImageColumn). API_URL is the in-cluster
# hostname (``iqa-api:8000``) used for server-side requests; the browser cannot resolve
# that, so image URLs must use the published host instead.
API_PUBLIC_URL = os.environ.get("IQA_API_PUBLIC_URL", "http://localhost:8000")


def get(path: str, *, timeout: int = 5) -> Any:
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def post(path: str, json: dict, *, headers: dict | None = None, timeout: int = 10) -> Any:
    response = requests.post(f"{API_URL}{path}", json=json, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()
