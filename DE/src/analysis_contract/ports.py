"""The boundary the DE/AI team implements. Adapters: Mock (M2) -> real pipeline (M5)."""

from collections.abc import Awaitable, Callable
from typing import Protocol

from .models import AnalysisRequest, AnalysisResult, JobProgress

ProgressCallback = Callable[[JobProgress], Awaitable[None]]


class AnalysisEngine(Protocol):
    async def analyze(
        self, request: AnalysisRequest, on_progress: ProgressCallback
    ) -> AnalysisResult:
        """Run the analysis. Report ``processing`` progress through ``on_progress``;
        raise ``AnalysisError`` (carrying a ``JobError``) on failure."""
        ...


class AnalysisError(Exception):
    def __init__(self, code, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
