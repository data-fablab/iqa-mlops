"""Contract tests for the IQA object-store data plane (ADR 0008 / KEN05).

The data-plane runtimes (ingestion, dataset, gates, mlflow) reach data,
manifests and model manifests through the ``ObjectStore`` interface, not through
a repo mount. These tests pin that contract: the in-memory double for offline
runs, the boto3/MinIO backend (stubbed, no live MinIO) and the env-driven
factory that mirrors ``create_metadata_repository``.
"""

from __future__ import annotations

import pytest

from iqa.storage.object_store import (
    MEMORY_BACKEND,
    S3_BACKEND,
    InMemoryObjectStore,
    ObjectStore,
    S3ObjectStore,
    create_object_store,
    object_store_backend,
)


def test_in_memory_round_trip_returns_s3_uri() -> None:
    store = InMemoryObjectStore()

    uri = store.put_bytes("iqa-models", "prod/model_manifest.json", b"{}")

    assert uri == "s3://iqa-models/prod/model_manifest.json"
    assert store.get_bytes("iqa-models", "prod/model_manifest.json") == b"{}"


def test_in_memory_get_missing_key_raises_key_error_with_uri() -> None:
    store = InMemoryObjectStore()

    with pytest.raises(KeyError, match="s3://iqa-dvc/data/raw/hss-iad.dvc"):
        store.get_bytes("iqa-dvc", "data/raw/hss-iad.dvc")


def test_in_memory_exists_reflects_presence() -> None:
    store = InMemoryObjectStore()

    assert store.exists("iqa-models", "candidate.pt") is False
    store.put_bytes("iqa-models", "candidate.pt", b"weights")
    assert store.exists("iqa-models", "candidate.pt") is True


def test_in_memory_stores_a_defensive_copy_of_the_body() -> None:
    store = InMemoryObjectStore()
    body = bytearray(b"manifest")

    store.put_bytes("iqa-source-datasets", "manifest.csv", body)
    body.extend(b"-mutated")

    assert store.get_bytes("iqa-source-datasets", "manifest.csv") == b"manifest"


def test_in_memory_can_be_seeded_from_constructor() -> None:
    store = InMemoryObjectStore({("iqa-dvc", "dvc.yaml"): b"stages: {}"})

    assert store.exists("iqa-dvc", "dvc.yaml") is True
    assert store.get_bytes("iqa-dvc", "dvc.yaml") == b"stages: {}"


def test_backend_defaults_to_memory(monkeypatch) -> None:
    monkeypatch.delenv("IQA_OBJECT_STORE_BACKEND", raising=False)

    assert object_store_backend() == MEMORY_BACKEND


def test_backend_reads_env_normalised(monkeypatch) -> None:
    monkeypatch.setenv("IQA_OBJECT_STORE_BACKEND", "  S3  ")

    assert object_store_backend() == S3_BACKEND


def test_factory_defaults_to_in_memory_store(monkeypatch) -> None:
    monkeypatch.delenv("IQA_OBJECT_STORE_BACKEND", raising=False)

    assert isinstance(create_object_store(), InMemoryObjectStore)


def test_factory_opts_into_s3_backend(monkeypatch) -> None:
    monkeypatch.setenv("IQA_OBJECT_STORE_BACKEND", "s3")
    monkeypatch.setenv("IQA_S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("IQA_S3_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("IQA_S3_SECRET_ACCESS_KEY", "secret")

    # No network: S3ObjectStore builds its boto3 client lazily, so construction
    # alone must not require a reachable MinIO.
    assert isinstance(create_object_store(), S3ObjectStore)


def test_factory_rejects_unknown_backend(monkeypatch) -> None:
    monkeypatch.setenv("IQA_OBJECT_STORE_BACKEND", "gcs")

    with pytest.raises(RuntimeError, match="IQA_OBJECT_STORE_BACKEND"):
        create_object_store()


def test_both_implementations_satisfy_the_protocol() -> None:
    assert isinstance(InMemoryObjectStore(), ObjectStore)
    assert isinstance(S3ObjectStore(), ObjectStore)


# --- S3 backend: behaviour against a stubbed boto3 client (no live MinIO) -----


def _stubbed_s3_store():
    """Return an ``S3ObjectStore`` and a ``Stubber`` wrapping its boto3 client."""
    from botocore.stub import Stubber

    store = S3ObjectStore(
        endpoint_url="http://minio:9000",
        access_key_id="key",
        secret_access_key="secret",
    )
    return store, Stubber(store._get_client())


def _streaming_body(data: bytes):
    import io

    from botocore.response import StreamingBody

    return StreamingBody(io.BytesIO(data), len(data))


def test_s3_get_bytes_returns_object_body() -> None:
    store, stubber = _stubbed_s3_store()
    stubber.add_response(
        "get_object",
        {"Body": _streaming_body(b"weights")},
        {"Bucket": "iqa-models", "Key": "candidate.pt"},
    )

    with stubber:
        assert store.get_bytes("iqa-models", "candidate.pt") == b"weights"


def test_s3_get_bytes_maps_missing_object_to_key_error() -> None:
    store, stubber = _stubbed_s3_store()
    stubber.add_client_error(
        "get_object", service_error_code="NoSuchKey", http_status_code=404
    )

    with stubber, pytest.raises(KeyError, match="s3://iqa-dvc/dvc.yaml"):
        store.get_bytes("iqa-dvc", "dvc.yaml")


def test_s3_exists_true_when_head_object_succeeds() -> None:
    store, stubber = _stubbed_s3_store()
    stubber.add_response(
        "head_object",
        {"ContentLength": 7},
        {"Bucket": "iqa-models", "Key": "candidate.pt"},
    )

    with stubber:
        assert store.exists("iqa-models", "candidate.pt") is True


def test_s3_exists_false_when_object_absent() -> None:
    store, stubber = _stubbed_s3_store()
    stubber.add_client_error(
        "head_object", service_error_code="404", http_status_code=404
    )

    with stubber:
        assert store.exists("iqa-models", "missing.pt") is False


def test_s3_put_bytes_sends_body_and_returns_uri() -> None:
    store, stubber = _stubbed_s3_store()
    stubber.add_response(
        "put_object",
        {},
        {"Bucket": "iqa-models", "Key": "candidate.pt", "Body": b"weights"},
    )

    with stubber:
        uri = store.put_bytes("iqa-models", "candidate.pt", b"weights")

    assert uri == "s3://iqa-models/candidate.pt"
    stubber.assert_no_pending_responses()
