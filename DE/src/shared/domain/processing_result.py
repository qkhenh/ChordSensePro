"""Railway-oriented result wrapper: ProcessingResult[T] with Ok and Err variants for controlled error handling."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

@dataclass
class ProcessingResult(Generic[T]):
    _value : T | None
    _error : str | None
    
    @classmethod
    def ok(cls, value: T) -> ProcessingResult[T]:
        return cls(_value=value, _error=None)
    
    @classmethod
    def err(cls, error: str) -> ProcessingResult[T]:
        return cls(_value=None, _error=error)
    
    @property
    def is_ok(self) -> bool:
        return self._error is None
    
    @property
    def is_err(self) -> bool:
        return self._error is not None
    
    @property
    def error(self) -> str:
        assert self._error is not None
        return self._error
    
    def unwrap(self) -> T:
        if self.is_err:
            raise AssertionError(f"unwrap() on Err: {self._error}")
        return self._value  # type: ignore[return-value]
    
    def unwrap_or(self, default: T) -> T:
        return self._value if self.is_ok else default  # type: ignore[return-value]
    
    def __repr__(self) -> str:
        return f"Ok({self._value!r})" if self.is_ok else f"Err({self._error!r})"
  
  
  
