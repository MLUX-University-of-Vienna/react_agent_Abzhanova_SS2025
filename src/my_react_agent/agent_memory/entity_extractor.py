from abc import ABC, abstractmethod
from typing import Optional, List, Any

from .data_structures import Entity


class EntityExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> Optional[Any]:
        pass

    def extract_all(self, text: str) -> List[Any]:
        one = self.extract(text)
        return [one] if one else []

    def extract_entities(self, text: str) -> List[str]:
        out: List[str] = []

        for obj in self.extract_all(text):
            if obj is None:
                continue

            if isinstance(obj, str):
                nm = obj.strip()
            else:
                nm = (
                    getattr(obj, "canonical_name", None)
                    or getattr(obj, "name", None)
                    or getattr(obj, "text", None)
                    or ""
                )
                nm = nm.strip() if isinstance(nm, str) else ""

            if nm:
                out.append(nm)

        seen = set()
        deduped: List[str] = []
        for n in out:
            k = n.lower()
            if k in seen:
                continue
            seen.add(k)
            deduped.append(n)
        return deduped
