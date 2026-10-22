"""NFR-09: expired, tampered or revoked tokens are rejected in 100% of cases."""

import base64
import json
from datetime import timedelta
from uuid import uuid4

import jwt
import pytest

from src.identity.application.tokens import TokenService
from src.identity.domain.errors import InvalidToken, TokenExpired, TokenRevoked
from src.identity.infrastructure.memory import InMemoryRevocationStore
from tests.support.identity import SECRET, FakeClock


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def service(clock):
    return TokenService(
        SECRET,
        access_ttl=timedelta(minutes=15),
        refresh_ttl=timedelta(days=14),
        revocations=InMemoryRevocationStore(),
        clock=clock,
    )


def b64(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


async def test_access_token_round_trip(service):
    uid = uuid4()
    pair = service.issue_pair(uid)
    claims = await service.verify(pair.access_token, "access")
    assert claims.user_id == uid
    assert pair.expires_in == 15 * 60
    assert pair.token_type == "bearer"


async def test_refresh_token_round_trip(service):
    pair = service.issue_pair(uuid4())
    assert (await service.verify(pair.refresh_token, "refresh")).token_type == "refresh"


async def test_expired_token_is_rejected(service, clock):
    pair = service.issue_pair(uuid4())
    clock.advance(minutes=15, seconds=1)
    with pytest.raises(TokenExpired):
        await service.verify(pair.access_token, "access")


async def test_token_is_still_valid_just_before_expiry(service, clock):
    pair = service.issue_pair(uuid4())
    clock.advance(minutes=14, seconds=59)
    assert await service.verify(pair.access_token, "access")


async def test_refresh_token_cannot_be_used_as_access_token_or_vice_versa(service):
    pair = service.issue_pair(uuid4())
    with pytest.raises(InvalidToken):
        await service.verify(pair.refresh_token, "access")
    with pytest.raises(InvalidToken):
        await service.verify(pair.access_token, "refresh")


async def test_tampered_payload_is_rejected(service):
    pair = service.issue_pair(uuid4())
    header, _payload, signature = pair.access_token.split(".")
    forged = b64(
        {"sub": str(uuid4()), "type": "access", "jti": "x", "iat": 0, "exp": 9_999_999_999}
    )
    with pytest.raises(InvalidToken):
        await service.verify(f"{header}.{forged}.{signature}", "access")


async def test_token_signed_with_another_secret_is_rejected(service, clock):
    now = int(clock().timestamp())
    foreign = jwt.encode(
        {"sub": str(uuid4()), "type": "access", "jti": "j", "iat": now, "exp": now + 600},
        "another-secret-" + "y" * 32,
        algorithm="HS256",
    )
    with pytest.raises(InvalidToken):
        await service.verify(foreign, "access")


async def test_unsigned_alg_none_token_is_rejected(service, clock):
    now = int(clock().timestamp())
    token = (
        b64({"alg": "none", "typ": "JWT"})
        + "."
        + b64({"sub": str(uuid4()), "type": "access", "jti": "j", "iat": now, "exp": now + 600})
        + "."
    )
    with pytest.raises(InvalidToken):
        await service.verify(token, "access")


async def test_token_with_a_different_signing_algorithm_is_rejected(service, clock):
    now = int(clock().timestamp())
    hs512 = jwt.encode(
        {"sub": str(uuid4()), "type": "access", "jti": "j", "iat": now, "exp": now + 600},
        SECRET,
        algorithm="HS512",
    )
    with pytest.raises(InvalidToken):
        await service.verify(hs512, "access")


@pytest.mark.parametrize("missing", ["sub", "exp", "jti", "type", "iat"])
async def test_token_missing_a_required_claim_is_rejected(service, clock, missing):
    now = int(clock().timestamp())
    claims = {"sub": str(uuid4()), "type": "access", "jti": "j", "iat": now, "exp": now + 600}
    del claims[missing]
    with pytest.raises(InvalidToken):
        await service.verify(jwt.encode(claims, SECRET, algorithm="HS256"), "access")


async def test_subject_must_be_a_uuid(service, clock):
    now = int(clock().timestamp())
    token = jwt.encode(
        {"sub": "not-a-uuid", "type": "access", "jti": "j", "iat": now, "exp": now + 600},
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(InvalidToken):
        await service.verify(token, "access")


@pytest.mark.parametrize("garbage", ["", "abc", "a.b", "a.b.c", "....", "Bearer x.y.z"])
async def test_garbage_is_rejected(service, garbage):
    with pytest.raises(InvalidToken):
        await service.verify(garbage, "access")


async def test_revoked_token_is_rejected(service):
    pair = service.issue_pair(uuid4())
    claims = await service.verify(pair.refresh_token, "refresh")
    await service.revoke(claims)
    with pytest.raises(TokenRevoked):
        await service.verify(pair.refresh_token, "refresh")


async def test_each_token_gets_a_unique_id(service):
    uid = uuid4()
    a, b = service.issue_pair(uid), service.issue_pair(uid)
    ids = {
        (await service.verify(t, kind)).jti
        for t, kind in [
            (a.access_token, "access"),
            (a.refresh_token, "refresh"),
            (b.access_token, "access"),
            (b.refresh_token, "refresh"),
        ]
    }
    assert len(ids) == 4


def test_a_short_signing_secret_is_refused():
    with pytest.raises(ValueError, match="secret"):
        TokenService(
            "short",
            access_ttl=timedelta(minutes=1),
            refresh_ttl=timedelta(days=1),
            revocations=InMemoryRevocationStore(),
        )
