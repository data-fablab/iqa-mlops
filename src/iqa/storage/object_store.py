"""Object-store client for the IQA MinIO/S3 data plane.

``iqa.storage.uris`` only models *where* objects live (bucket + key). This module
adds the *how*: a tiny ``ObjectStore`` protocol with two implementations:

- ``S3ObjectStore`` -- boto3 against MinIO/S3 (lazy import: boto3 is a ``data``/
  ``serving`` role dependency, never pulled just by importing this module).
- ``InMemoryObjectStore`` -- a dict-backed double for tests and dry runs, so the
  ingestion/dataset runtimes are exercised without a live MinIO.

Backend and credentials come from the same env contract as ``.env.example``
(``IQA_S3_ENDPOINT_URL`` / ``IQA_S3_ACCESS_KEY_ID`` / ``IQA_S3_SECRET_ACCESS_KEY``
/ ``IQA_S3_REGION``). ``create_object_store()`` mirrors
``create_metadata_repository()``: ``memory`` by default (safe, offline), ``s3``
when the container opts in via ``IQA_OBJECT_STORE_BACKEND=s3``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime boto3 import.
    pass


OBJECT_STORE_BACKEND_ENV = "IQA_OBJECT_STORE_BACKEND"
S3_ENDPOINT_URL_ENV = "IQA_S3_ENDPOINT_URL"
S3_ACCESS_KEY_ID_ENV = "IQA_S3_ACCESS_KEY_ID"
S3_SECRET_ACCESS_KEY_ENV = "IQA_S3_SECRET_ACCESS_KEY"
S3_REGION_ENV = "IQA_S3_REGION"
MEMORY_BACKEND = "memory"
S3_BACKEND = "s3"


@runtime_checkable
class ObjectStore(Protocol):
    """Minimal object-store contract used by the IQA data-plane runtimes."""

    def get_bytes(self, bucket: str, key: str) -> bytes:
        """Return the object body, or raise ``KeyError`` if it is absent."""

    def put_bytes(self, bucket: str, key: str, data: bytes, *, content_type: str | None = None) -> str:
        """Store ``data`` and return the ``s3://`` URI of the written object."""

    def exists(self, bucket: str, key: str) -> bool:
        """Return whether an object exists at ``bucket``/``key``."""


def _s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


class InMemoryObjectStore:
    """Dict-backed object store for tests and dry runs (no network)."""

    def __init__(self, objects: dict[tuple[str, str], bytes] | None = None) -> None:
        self._objects: dict[tuple[str, str], bytes] = dict(objects or {})

    def get_bytes(self, bucket: str, key: str) -> bytes:
        try:
            return self._objects[(bucket, key)]
        except KeyError:
            raise KeyError(_s3_uri(bucket, key)) from None

    def put_bytes(self, bucket: str, key: str, data: bytes, *, content_type: str | None = None) -> str:
        _ = content_type  # accepted for parity with S3; irrelevant in memory.
        self._objects[(bucket, key)] = bytes(data)
        return _s3_uri(bucket, key)

    def exists(self, bucket: str, key: str) -> bool:
        return (bucket, key) in self._objects


class S3ObjectStore:
    """boto3-backed object store against MinIO/S3 (lazy client construction)."""

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region: str | None = None,
    ) -> None:
        self._endpoint_url = endpoint_url or os.getenv(S3_ENDPOINT_URL_ENV)
        self._access_key_id = access_key_id or os.getenv(S3_ACCESS_KEY_ID_ENV)
        self._secret_access_key = secret_access_key or os.getenv(S3_SECRET_ACCESS_KEY_ENV)
        self._region = region or os.getenv(S3_REGION_ENV, "us-east-1")
        self._client = None

    def _get_client(self):  # noqa: ANN202 - boto3 client type is dynamic.
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
                region_name=self._region,
            )
        return self._client

    def get_bytes(self, bucket: str, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            response = self._get_client().get_object(Bucket=bucket, Key=key)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code in {"NoSuchKey", "404", "NoSuchBucket"}:
                raise KeyError(_s3_uri(bucket, key)) from error
            raise
        return response["Body"].read()

    def put_bytes(self, bucket: str, key: str, data: bytes, *, content_type: str | None = None) -> str:
        extra = {"ContentType": content_type} if content_type else {}
        self._get_client().put_object(Bucket=bucket, Key=key, Body=data, **extra)
        return _s3_uri(bucket, key)

    def exists(self, bucket: str, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._get_client().head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code in {"NoSuchKey", "404", "NoSuchBucket"}:
                return False
            raise
        return True


def object_store_backend() -> str:
    """Return the configured object-store backend name."""
    return os.getenv(OBJECT_STORE_BACKEND_ENV, MEMORY_BACKEND).strip().lower()


def create_object_store() -> ObjectStore:
    """Create the configured object store (``memory`` default, ``s3`` opt-in)."""
    backend = object_store_backend()
    if backend == MEMORY_BACKEND:
        return InMemoryObjectStore()
    if backend == S3_BACKEND:
        return S3ObjectStore()
    raise RuntimeError(
        f"Unsupported {OBJECT_STORE_BACKEND_ENV}: {backend!r}. Expected 'memory' or 's3'."
    )


__all__ = [
    "InMemoryObjectStore",
    "ObjectStore",
    "OBJECT_STORE_BACKEND_ENV",
    "S3ObjectStore",
    "S3_BACKEND",
    "MEMORY_BACKEND",
    "create_object_store",
    "object_store_backend",
]
