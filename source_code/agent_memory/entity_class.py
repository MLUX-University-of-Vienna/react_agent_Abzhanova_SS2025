from typing import Optional
from dataclasses import dataclass, field
from .gender_enum import GenderEnum

@dataclass
class Entity:
    text: str
    label: Optional[str] = None     
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    gender: GenderEnum = GenderEnum.UNKNOWN
    age: Optional[int] = None       

    def possessive(self) -> str:
        if self.text.endswith("s"):
            return f"{self.text}'"
        return f"{self.text}'s"