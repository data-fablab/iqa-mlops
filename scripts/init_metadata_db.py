"""Initialize the IQA PostgreSQL metadata schema."""

from __future__ import annotations

import argparse

from iqa.metadata.repository import METADATA_DB_URL_ENV, metadata_db_url
from iqa.metadata.postgres import initialize_metadata_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url",
        default=None,
        help=f"PostgreSQL URL. Defaults to ${METADATA_DB_URL_ENV}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_url = args.db_url or metadata_db_url()
    if not db_url:
        raise SystemExit(f"{METADATA_DB_URL_ENV} is required. Set it or pass --db-url.")

    initialize_metadata_db(db_url)
    print("IQA metadata schema is initialized.")


if __name__ == "__main__":
    main()
