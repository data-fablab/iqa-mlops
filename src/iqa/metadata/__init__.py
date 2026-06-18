"""IQA metadata repository foundation.

``iqa.metadata.contracts`` pulls pandas (a ``data`` role dependency). It is
imported lazily (PEP 562) so that ``serving`` consumers that only need the
repository (e.g. the API gateway) do not pull pandas through this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing/re-export only, never executed at runtime.
    from iqa.metadata.contracts import MANIFEST_CONTRACTS as MANIFEST_CONTRACTS
    from iqa.metadata.contracts import PHASE2_METADATA_COLUMNS as PHASE2_METADATA_COLUMNS
    from iqa.metadata.contracts import RAW_DATASET_ID as RAW_DATASET_ID
    from iqa.metadata.contracts import MetadataManifestContract as MetadataManifestContract
    from iqa.metadata.contracts import apply_metadata_contract as apply_metadata_contract
    from iqa.metadata.contracts import contract_for_key as contract_for_key
    from iqa.metadata.repository import MEMORY_BACKEND as MEMORY_BACKEND
    from iqa.metadata.repository import METADATA_BACKEND_ENV as METADATA_BACKEND_ENV
    from iqa.metadata.repository import METADATA_DB_URL_ENV as METADATA_DB_URL_ENV
    from iqa.metadata.repository import POSTGRES_BACKEND as POSTGRES_BACKEND
    from iqa.metadata.repository import MemoryMetadataRepository as MemoryMetadataRepository
    from iqa.metadata.repository import MetadataRepository as MetadataRepository
    from iqa.metadata.repository import create_metadata_repository as create_metadata_repository
    from iqa.metadata.repository import metadata_backend as metadata_backend
    from iqa.metadata.repository import metadata_db_url as metadata_db_url

# Public attribute -> defining submodule. Resolved on first access only.
_LAZY_EXPORTS = {
    "MANIFEST_CONTRACTS": "iqa.metadata.contracts",
    "PHASE2_METADATA_COLUMNS": "iqa.metadata.contracts",
    "RAW_DATASET_ID": "iqa.metadata.contracts",
    "MetadataManifestContract": "iqa.metadata.contracts",
    "apply_metadata_contract": "iqa.metadata.contracts",
    "contract_for_key": "iqa.metadata.contracts",
    "MEMORY_BACKEND": "iqa.metadata.repository",
    "METADATA_BACKEND_ENV": "iqa.metadata.repository",
    "METADATA_DB_URL_ENV": "iqa.metadata.repository",
    "POSTGRES_BACKEND": "iqa.metadata.repository",
    "MemoryMetadataRepository": "iqa.metadata.repository",
    "MetadataRepository": "iqa.metadata.repository",
    "create_metadata_repository": "iqa.metadata.repository",
    "metadata_backend": "iqa.metadata.repository",
    "metadata_db_url": "iqa.metadata.repository",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return [*globals(), *_LAZY_EXPORTS]
