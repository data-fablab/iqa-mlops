from pathlib import Path


KONG_CONFIG = Path("deploy/kong/kong.yml")


def read_kong_config() -> str:
    assert KONG_CONFIG.exists()
    return KONG_CONFIG.read_text(encoding="utf-8")


def test_kong_config_exists_and_declares_db_less_format():
    text = read_kong_config()

    assert '_format_version: "3.0"' in text
    assert "_transform: true" in text
    assert "consumers:" in text
    assert "services:" in text


def test_kong_consumers_and_acl_groups_are_defined():
    text = read_kong_config()

    assert "username: iqa-service" in text
    assert "username: iqa-admin" in text
    assert "username: iqa-monitoring" in text
    assert "group: service" in text
    assert "group: admin" in text
    assert "group: monitoring" in text


def test_sensitive_api_routes_require_gateway_auth():
    text = read_kong_config()

    for route in [
        "/api/admin/reload-model",
        "/api/metrics",
        "/api",
    ]:
        assert route in text

    assert "name: key-auth" in text
    assert "X-IQA-Gateway-Key" in text
    assert "hide_credentials: true" in text


def test_admin_reload_route_is_strictly_protected():
    text = read_kong_config()

    start = text.index("name: iqa-api-admin-reload")
    end = text.index("name: iqa-api-metrics")
    block = text[start:end]

    assert "/api/admin/reload-model" in block
    assert "name: key-auth" in block
    assert "name: acl" in block
    assert "admin" in block
    assert "name: rate-limiting" in block
    assert "minute: 3" in block
    assert "name: request-size-limiting" in block
    assert "allowed_payload_size: 1" in block


def test_metrics_route_is_restricted_to_monitoring_or_admin():
    text = read_kong_config()

    start = text.index("name: iqa-api-metrics")
    end = text.index("name: iqa-api-protected")
    block = text[start:end]

    assert "/api/metrics" in block
    assert "name: key-auth" in block
    assert "name: acl" in block
    assert "monitoring" in block
    assert "admin" in block
    assert "name: rate-limiting" in block


def test_platform_admin_routes_are_not_public():
    text = read_kong_config()

    for route_name in [
        "name: iqa-mlflow-ui",
        "name: iqa-minio-ui",
        "name: iqa-airflow-ui",
    ]:
        start = text.index(route_name)
        block = text[start:start + 900]
        assert "name: key-auth" in block
        assert "name: acl" in block
        assert "admin" in block


def test_security_headers_are_declared_at_gateway_level():
    text = read_kong_config()

    assert "name: response-transformer" in text
    assert "X-Content-Type-Options:nosniff" in text
    assert "Referrer-Policy:no-referrer" in text
    assert "X-Frame-Options:DENY" in text
