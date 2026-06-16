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
    MemoryMetadataRepository,
    MetadataRepository,
    metadata_db_url,
)

__all__ = [
    "MANIFEST_CONTRACTS",
    "PHASE2_METADATA_COLUMNS",
    "RAW_DATASET_ID",
    "MemoryMetadataRepository",
    "MetadataManifestContract",
    "MetadataRepository",
    "apply_metadata_contract",
    "contract_for_key",
    "metadata_db_url",
]
