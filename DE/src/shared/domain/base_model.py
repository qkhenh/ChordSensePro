"""Base entity classes for all domain models: BaseEntity (UUID + created_at) and AuditedEntity (+ updated_at)."""

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
