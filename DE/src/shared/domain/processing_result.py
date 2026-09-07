"""
shared/domain/processing_result.py

TODO: Bạn sẽ viết generic result type cho pipeline.

Tại sao không dùng Exception để báo lỗi pipeline?
  - Exception = dành cho lỗi KHÔNG lường trước (bug, crash)
  - ProcessingResult = lỗi CÓ kiểm soát (file không tồn tại, audio quá ngắn...)

Pattern này gọi là "Railway-oriented programming":
  Ok(value)  → step thành công → pass value sang bước tiếp
  Err(msg)   → step thất bại  → dừng lại, log lỗi, Airflow retry

Bạn cần viết:
1. Class ProcessingResult[T] (Generic)
   - classmethod ok(value)  → trả về Ok result
   - classmethod err(msg)   → trả về Err result
   - property is_ok / is_err
   - method unwrap()        → lấy value, raise nếu Err
   - method unwrap_or(default) → lấy value hoặc default
"""

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
        if self._value is None:
            raise AssertionError(f"unwrap() trên Err: {self._error}")
        return self._value
    
    def unwrap_or(self, default: T) -> T:
        return self._value if self.is_ok else default
    
    def __repr__(self) -> str:
        return f"Ok({self._value!r})" if self.is_ok else f"Err({self._error!r})"
  
  
  
