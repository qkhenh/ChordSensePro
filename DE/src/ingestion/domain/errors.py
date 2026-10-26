"""Errors of the ingestion context; ``IngestionError`` carries a contract ``ErrorCode``."""

from src.analysis_contract.models import ErrorCode


class IngestionError(Exception):
    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class InvalidTransition(ValueError):
    """A job state change the lifecycle does not allow."""


class DemoNotFound(LookupError):
    """The id is not in the demo catalog."""


class DemoUnavailable(RuntimeError):
    """The demo audio has not been generated yet (run scripts/gen_demo_audio.py)."""
