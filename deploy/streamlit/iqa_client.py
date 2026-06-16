"""Shared IQA API helpers for the Streamlit Sophie/Marc views."""

from __future__ import annotations

import os
from typing import Any

import requests

API_URL = os.environ.get("IQA_API_URL", "http://localhost:8000")


def get(path: str, *, timeout: int = 5) -> Any:
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def post(path: str, json: dict, *, headers: dict | None = None, timeout: int = 10) -> Any:
    response = requests.post(f"{API_URL}{path}", json=json, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()
