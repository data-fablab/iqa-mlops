"""Run the IQA inference service."""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("iqa.inference.service:app", host="0.0.0.0", port=8100)


if __name__ == "__main__":
    main()
