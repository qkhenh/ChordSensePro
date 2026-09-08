"""BaseProcessingHandler — abstract Chain of Responsibility base class."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_processing.domain.models.processed_audio import ProcessedAudio


class BaseProcessingHandler(ABC):
    """Base class for all pipeline handlers (Chain of Responsibility pattern).

    Chain setup:
        separate = SeparateHandler()
        beat     = BeatHandler()
        segment  = SegmentHandler()
        feature  = FeatureHandler()
        separate.set_next(beat).set_next(segment).set_next(feature)
        result = separate.handle(data)

    Each handler processes data then calls _call_next() to forward downstream.
    If data.is_failed, handlers should short-circuit and return immediately.
    """

    def __init__(self) -> None:
        self._next: BaseProcessingHandler | None = None  # instance-level, not class-level

    def set_next(self, handler: BaseProcessingHandler) -> BaseProcessingHandler:
        """Chain this handler to the next. Returns next for fluent chaining."""
        self._next = handler
        return handler

    @abstractmethod
    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        """Process data and return (possibly enriched) ProcessedAudio."""
        ...

    def _call_next(self, data: ProcessedAudio) -> ProcessedAudio:
        """Forward to next handler if set, otherwise return data as-is."""
        if self._next:
            return self._next.handle(data)
        return data
