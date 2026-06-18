"""Run the IQA API service."""

from __future__ import annotations

import argparse

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Hot-reload on code change (dev only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run("iqa.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
