"""
shared/domain/base_model.py

TODO: Bạn sẽ viết base classes cho tất cả Domain Entities.

Phân biệt Entity vs Value Object:
  - Entity   = CÓ ID riêng, mutable, so sánh bằng ID
  - Value Object = KHÔNG có ID, immutable, so sánh bằng values

Bạn cần viết:
1. BaseEntity   — có id (UUID string) + created_at (UTC)
2. AuditedEntity — kế thừa BaseEntity, thêm updated_at + method touch()
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class BaseEntity:
    id : str = field(default_factory=lambda : str(uuid.uuid4()))
    created_at : datetime = field(default_factory=lambda : datetime.now(timezone.utc))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__): return False
        return self.id == other.id
        
    def __hash__(self) -> int:
        return hash(self.id)
    
@dataclass
class AuditedEntity(BaseEntity):
    updated_at : datetime = field(default_factory=lambda : datetime.now(timezone.utc))
    
    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
