from datetime import timedelta
from uuid import uuid4

import jwt
import pytest

from src.identity.application.media_tickets import MediaTicketService
from src.identity.domain.errors import InvalidToken, TokenExpired
from tests.support.identity import SECRET, FakeClock, make_container

TTL = timedelta(minutes=30)


def service(clock: FakeClock | None = None) -> tuple[MediaTicketService, FakeClock]:
    clock = clock or FakeClock()
    return MediaTicketService(SECRET, ttl=TTL, clock=clock), clock


def test_a_ticket_names_its_user_and_expires_after_the_ttl():
    tickets, clock = service()
    user, chart = uuid4(), uuid4()
    token, expires_at = tickets.issue(user, chart)
    assert expires_at == clock() + TTL
    assert tickets.verify(token, chart) == user


def test_it_is_good_until_the_last_second_and_refused_after():
    tickets, clock = service()
    user, chart = uuid4(), uuid4()
    token, _ = tickets.issue(user, chart)
    clock.advance(minutes=29, seconds=59)
    assert tickets.verify(token, chart) == user
    clock.advance(seconds=2)
    with pytest.raises(TokenExpired):
        tickets.verify(token, chart)


def test_it_opens_one_chart_only():
    tickets, _ = service()
    token, _ = tickets.issue(uuid4(), uuid4())
    with pytest.raises(InvalidToken):
        tickets.verify(token, uuid4())


@pytest.mark.parametrize("garbage", ["", "abc", "a.b.c", "x" * 400])
def test_garbage_is_invalid(garbage):
    tickets, _ = service()
    with pytest.raises(InvalidToken):
        tickets.verify(garbage, uuid4())


def test_a_token_signed_with_another_secret_or_without_one_is_invalid():
    tickets, clock = service()
    chart, user = uuid4(), uuid4()
    now = int(clock().timestamp())
    claims = {
        "sub": str(user),
        "chart": str(chart),
        "type": "media",
        "jti": "j",
        "iat": now,
        "exp": now + 600,
    }
    forged = jwt.encode(claims, "another-secret-" + "x" * 48, algorithm="HS256")
    unsigned = jwt.encode(claims, None, algorithm="none")
    for token in (forged, unsigned):
        with pytest.raises(InvalidToken):
            tickets.verify(token, chart)


def test_a_ticket_without_a_chart_claim_is_invalid():
    tickets, clock = service()
    now = int(clock().timestamp())
    token = jwt.encode(
        {"sub": str(uuid4()), "type": "media", "jti": "j", "iat": now, "exp": now + 600},
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(InvalidToken):
        tickets.verify(token, uuid4())


async def test_session_tokens_are_not_tickets_and_tickets_are_not_session_tokens():
    container = make_container()
    tickets, _ = service(FakeClock())
    user, chart = uuid4(), uuid4()
    pair = container.tokens.issue_pair(user)

    for session_token in (pair.access_token, pair.refresh_token):
        with pytest.raises(InvalidToken):
            tickets.verify(session_token, chart)

    ticket, _ = tickets.issue(user, chart)
    for kind in ("access", "refresh"):
        with pytest.raises(InvalidToken):
            await container.tokens.verify(ticket, kind)
