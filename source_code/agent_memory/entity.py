from dataclasses import dataclass, field
from typing import Optional, Union, Any, Dict, List, Set
from datetime import datetime
from .gender_enum import GenderEnum

Value = Union[str, int, float, Dict[str, Any]]

@dataclass(frozen=True)
class Fact:
    entity_id: str
    predicate: str
    value: Value
    as_of: Optional[datetime] = None
    source: str = "unknown"
    confidence: float = 0.8

@dataclass(frozen=True)
class Entity:
    entity_id: str
    canonical_name: str
    aliases: Set[str] = field(default_factory=set)
    gender: GenderEnum = GenderEnum.UNKNOWN

@dataclass(frozen=True)
class Mention:
    text: str
    start: int
    end: int
    gender: GenderEnum = GenderEnum.UNKNOWN
    entity_id: Optional[str] = None
