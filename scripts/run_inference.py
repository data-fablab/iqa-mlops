"""Run the IQA inference service."""

from __future__ import annotations

import argparse

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run("iqa.inference.service:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
