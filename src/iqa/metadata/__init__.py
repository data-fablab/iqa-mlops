"""IQA metadata repository foundation."""

from iqa.metadata.contracts import (
    MANIFEST_CONTRACTS,
    PHASE2_METADATA_COLUMNS,
    RAW_DATASET_ID,
    MetadataManifestContract,
    apply_metadata_contract,
    contract_for_key,
)
from iqa.metadata.repository import (
    MEMORY_BACKEND,
    METADATA_BACKEND_ENV,
    METADATA_DB_URL_ENV,
    POSTGRES_BACKEND,
    MemoryMetadataRepository,
    MetadataRepository,
    create_metadata_repository,
    metadata_backend,
    metadata_db_url,
)

__all__ = [
    "MANIFEST_CONTRACTS",
    "PHASE2_METADATA_COLUMNS",
    "RAW_DATASET_ID",
    "MEMORY_BACKEND",
    "METADATA_BACKEND_ENV",
    "METADATA_DB_URL_ENV",
    "MemoryMetadataRepository",
    "MetadataManifestContract",
    "MetadataRepository",
    "POSTGRES_BACKEND",
    "apply_metadata_contract",
    "contract_for_key",
    "create_metadata_repository",
    "metadata_backend",
    "metadata_db_url",
]
