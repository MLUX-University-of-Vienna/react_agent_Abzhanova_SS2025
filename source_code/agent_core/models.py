from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from .enums import QueryStatus, SubqueryStatus, ExecutionMode
from datetime import datetime

@dataclass
class ToolCall:
    name: str
    input: str
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class Subquery:
    id: str
    text: str
    mode: ExecutionMode
    status: SubqueryStatus = SubqueryStatus.PLANNED
    tool_calls: List[ToolCall] = field(default_factory=list)
    answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Query:
    id: str
    original_text: str
    status: QueryStatus = QueryStatus.PLANNED
    subqueries: List[Subquery] = field(default_factory=list)
    final_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolInput:
    query: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    # Core, typed
    value: Optional[float] = None  # use float/Decimal/str depending on your needs
    unit: Optional[str] = None  # "people", "USD", "km", etc.
    approx: bool = False

    # Provenance
    observed_at: Optional[datetime] = None
    source: Optional[str] = None  # "wikipedia", "serpapi", "calculator", "vector_search"
    source_url: Optional[str] = None

    # Human context
    raw_text: Optional[str] = None  # the exact snippet/sentence/infobox row used
    title: Optional[str] = None  # e.g., page title, result title

    # Freeform
    meta: Dict[str, Any] = field(default_factory=dict)

    # Convenience for legacy text-only paths
    def to_line(self) -> str:
        parts = []
        if self.value is not None:
            v = f"{self.value}"
            if self.unit:
                v += f" {self.unit}"
            parts.append(v)
        if self.observed_at:
            parts.append(f"as of {self.observed_at.date()}")
        if self.source_url:
            parts.append(f"src: {self.source_url}")
        if not parts and self.raw_text:
            parts.append(self.raw_text.strip())
        return " | ".join(parts) if parts else ""


@dataclass
class ToolResponse:
    success: bool
    observations: List[Observation] = field(default_factory=list)
    # optional human message for logs/debug
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- Back-compat helpers (so older code won’t crash) ---
    @property
    def content(self) -> str:
        """Best-effort human-readable content (joined lines)."""
        lines = [obs.to_line() or (obs.raw_text or "") for obs in self.observations]
        txt = " ; ".join([l for l in lines if l])
        return txt or (self.message or "")

    @property
    def primary_observation(self) -> Optional[Observation]:
        return self.observations[0] if self.observations else None
