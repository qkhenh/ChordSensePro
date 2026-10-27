"""``create_app``: health probe, uniform error bodies, request ids and CORS (task 1.4)."""

import httpx
import pytest
import structlog

from src.api.app import create_app
from src.api.settings import ApiSettings
from tests.support.identity import PASSWORD

ORIGIN = "http://localhost:5173"


@pytest.fixture
def app(container):
    app = create_app(container, ApiSettings(cors_origins=[ORIGIN]))

    @app.get("/_t/boom")
    async def boom():
        raise RuntimeError("secret internal detail")

    return app


@pytest.fixture
async def http(app):
    # Unhandled errors must surface as a 500 response, not re-raise into the test.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealth:
    async def test_healthz_reports_ok(self, http):
        r = await http.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestRouting:
    async def test_auth_and_users_are_mounted_under_api_v1(self, http):
        r = await http.post(
            "/api/v1/auth/register",
            json={"email": "ann@example.com", "password": PASSWORD, "name": "Ann"},
        )
        assert r.status_code == 201
        token = r.json()["access_token"]

        me = await http.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["email"] == "ann@example.com"


class TestErrorBodies:
    async def test_validation_error_has_code_message_and_field_detail(self, http):
        r = await http.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "password": PASSWORD, "name": "Ann"},
        )
        body = r.json()
        assert r.status_code == 422
        assert body["code"] == "validation_error"
        assert isinstance(body["message"], str) and body["message"]
        assert [d["field"] for d in body["detail"]] == ["email"]
        assert all(d["message"] for d in body["detail"])

    async def test_unknown_route_is_404_not_found(self, http):
        r = await http.get("/api/v1/nope")
        assert r.status_code == 404
        assert r.json()["code"] == "not_found"

    async def test_wrong_method_is_405_with_allow_header(self, http):
        r = await http.get("/api/v1/auth/login")
        assert r.status_code == 405
        assert r.json()["code"] == "method_not_allowed"
        assert "POST" in r.headers["allow"]

    async def test_unhandled_exception_is_500_without_leaking_details(self, http):
        r = await http.get("/_t/boom")
        assert r.status_code == 500
        assert r.json() == {"code": "internal_error", "message": "Internal server error"}
        assert "secret" not in r.text

    async def test_domain_errors_keep_their_code(self, http):
        r = await http.post(
            "/api/v1/auth/login", json={"email": "x@example.com", "password": "y" * 8}
        )
        assert r.status_code == 401
        assert r.json()["code"] == "invalid_credentials"


class TestRequestId:
    async def test_generates_an_id_when_the_client_sends_none(self, http):
        r = await http.get("/healthz")
        assert len(r.headers["x-request-id"]) >= 16

    async def test_echoes_a_well_formed_client_id(self, http):
        r = await http.get("/healthz", headers={"X-Request-ID": "req-123_ABC"})
        assert r.headers["x-request-id"] == "req-123_ABC"

    async def test_replaces_a_malformed_client_id(self, http):
        r = await http.get("/healthz", headers={"X-Request-ID": "bad id\twith spaces" + "x" * 200})
        assert r.headers["x-request-id"] != "bad id\twith spaces" + "x" * 200
        assert " " not in r.headers["x-request-id"]

    async def test_error_responses_carry_the_id_too(self, http):
        for path in ("/_t/boom", "/api/v1/nope"):
            r = await http.get(path, headers={"X-Request-ID": "trace-1"})
            assert r.headers["x-request-id"] == "trace-1", path

    async def test_each_request_is_logged_with_its_id_and_status(self, http):
        with structlog.testing.capture_logs() as logs:
            r = await http.get("/healthz", headers={"X-Request-ID": "trace-2"})
        entries = [e for e in logs if e["event"] == "request"]
        assert len(entries) == 1
        assert entries[0]["request_id"] == "trace-2"
        assert (entries[0]["method"], entries[0]["path"], entries[0]["status"]) == (
            "GET",
            "/healthz",
            r.status_code,
        )


class TestCors:
    async def test_allowed_origin_gets_preflight_approval(self, http):
        r = await http.options(
            "/api/v1/auth/login",
            headers={
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == ORIGIN

    async def test_other_origins_are_not_approved(self, http):
        r = await http.get("/healthz", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in r.headers

    async def test_request_id_is_readable_by_the_browser(self, http):
        r = await http.get("/healthz", headers={"Origin": ORIGIN})
        assert "x-request-id" in r.headers["access-control-expose-headers"].lower()

    async def test_range_and_validator_headers_are_readable_by_a_cross_origin_player(self, http):
        r = await http.get("/healthz", headers={"Origin": ORIGIN})
        exposed = r.headers["access-control-expose-headers"].lower()
        assert all(h in exposed for h in ("content-range", "accept-ranges", "etag"))

    async def test_a_cross_origin_page_may_revalidate_with_if_none_match(self, http):
        r = await http.options(
            "/api/v1/charts/00000000-0000-4000-8000-000000000001",
            headers={
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization,if-none-match",
            },
        )
        assert r.status_code == 200
        assert "if-none-match" in r.headers["access-control-allow-headers"].lower()


class TestSettings:
    def test_defaults_allow_the_vite_dev_server(self):
        assert "http://localhost:5173" in ApiSettings().cors_origins

    def test_wildcard_origin_is_refused(self):
        with pytest.raises(ValueError, match="wildcard"):
            ApiSettings(cors_origins=["*"])
