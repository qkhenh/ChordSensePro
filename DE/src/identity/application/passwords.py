"""bcrypt password hashing. CPU-bound: call through ``asyncio.to_thread`` from async code."""

import bcrypt

MAX_PASSWORD_BYTES = 72  # bcrypt ignores everything past this; refuse instead of truncating
DEFAULT_COST = 12


class PasswordHasher:
    def __init__(self, cost: int = DEFAULT_COST) -> None:
        self._cost = cost

    def hash(self, password: str) -> str:
        raw = password.encode()
        if len(raw) > MAX_PASSWORD_BYTES:
            raise ValueError(f"password must be at most {MAX_PASSWORD_BYTES} bytes")
        return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=self._cost)).decode()

    def verify(self, password: str, digest: str) -> bool:
        raw = password.encode()
        if len(raw) > MAX_PASSWORD_BYTES:
            return False
        try:
            return bcrypt.checkpw(raw, digest.encode())
        except ValueError:  # malformed digest
            return False
