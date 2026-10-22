from typing import ClassVar

import pytest
from pydantic import ValidationError

from src.identity.application.passwords import PasswordHasher
from src.identity.settings import AuthSettings


class TestPasswordHasher:
    def test_hash_is_not_the_plaintext_and_verifies(self):
        hasher = PasswordHasher(cost=4)
        digest = hasher.hash("s3cret-pass")
        assert digest != "s3cret-pass"
        assert hasher.verify("s3cret-pass", digest)
        assert not hasher.verify("wrong", digest)

    def test_each_hash_has_its_own_salt(self):
        hasher = PasswordHasher(cost=4)
        assert hasher.hash("same-password") != hasher.hash("same-password")

    def test_cost_factor_is_embedded_in_the_hash(self):
        assert PasswordHasher(cost=4).hash("pw-12345678").startswith("$2b$04$")

    def test_default_cost_meets_the_security_requirement(self):
        assert PasswordHasher().hash("pw-12345678").startswith("$2b$12$")  # NFR-09: cost >= 12

    def test_verify_never_raises_on_a_garbage_digest(self):
        assert PasswordHasher(cost=4).verify("pw", "not-a-bcrypt-hash") is False

    def test_passwords_over_72_bytes_are_refused_instead_of_silently_truncated(self):
        with pytest.raises(ValueError, match="72"):
            PasswordHasher(cost=4).hash("a" * 73)

    def test_multibyte_passwords_are_measured_in_bytes(self):
        with pytest.raises(ValueError, match="72"):
            PasswordHasher(cost=4).hash("ê" * 40)  # 80 bytes


class TestAuthSettings:
    GOOD: ClassVar[dict] = {"jwt_secret": "s" * 32}

    def test_defaults(self):
        s = AuthSettings(**self.GOOD)
        assert s.bcrypt_cost == 12
        assert s.access_ttl_minutes == 15
        assert s.refresh_ttl_days == 14

    def test_bcrypt_cost_below_twelve_is_rejected(self):
        with pytest.raises(ValidationError, match="bcrypt_cost"):
            AuthSettings(**self.GOOD, bcrypt_cost=10)

    def test_short_signing_secret_is_rejected(self):
        with pytest.raises(ValidationError, match="jwt_secret"):
            AuthSettings(jwt_secret="short")

    def test_ttls_must_be_positive(self):
        with pytest.raises(ValidationError):
            AuthSettings(**self.GOOD, access_ttl_minutes=0)

    def test_secret_is_not_leaked_by_repr(self):
        assert "s" * 32 not in repr(AuthSettings(**self.GOOD))
