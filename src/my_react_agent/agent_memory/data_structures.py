from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

JSONScalar = Union[str, int, float, bool, None]
JSONValue = Union[JSONScalar, Dict[str, "JSONValue"], List["JSONValue"]]

@dataclass(frozen=True)
class BaseId:
    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class QueryId(BaseId):
    @staticmethod
    def generate() -> "QueryId":
        return QueryId(f"qry_{uuid4().hex}")


@dataclass(frozen=True)
class StepId(BaseId):
    @staticmethod
    def generate() -> "StepId":
        return StepId(f"step_{uuid4().hex}")


@dataclass(frozen=True)
class EntityId(BaseId):
    @staticmethod
    def generate() -> "EntityId":
        return EntityId(f"ent_{uuid4().hex}")


@dataclass(frozen=True)
class MentionId(BaseId):
    @staticmethod
    def generate() -> "MentionId":
        return MentionId(f"mnt_{uuid4().hex}")


@dataclass(frozen=True)
class ToolCallId(BaseId):
    @staticmethod
    def generate() -> "ToolCallId":
        return ToolCallId(f"tc_{uuid4().hex}")


@dataclass(frozen=True)
class EvidenceExtractedFact:
    text: str


class GenderEnum(Enum):
    MALE = auto()
    FEMALE = auto()
    NEUTRAL = auto()
    UNKNOWN = auto()

    def __str__(self) -> str:
        return {
            GenderEnum.MALE: "male",
            GenderEnum.FEMALE: "female",
            GenderEnum.NEUTRAL: "neutral",
            GenderEnum.UNKNOWN: "unknown",
        }[self]


class QueryStatus(Enum):
    PLANNED = auto()
    RUNNING = auto()
    ANSWERED = auto()
    FAILED = auto()


class StepStatus(Enum):
    PLANNED = auto()
    RUNNING = auto()
    DONE = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass(frozen=True)
class EvidenceExtract:
    fields: Dict[str, JSONScalar] = field(default_factory=dict)
    payload: Optional[JSONValue] = None


@dataclass(frozen=True)
class Evidence:
    tool: str
    content: str
    url: Optional[str] = None
    extracted: EvidenceExtract = field(default_factory=EvidenceExtract)
    as_of: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.7


@dataclass(frozen=True)
class EntityRef:
    entity_id: EntityId
    canonical_name: str


@dataclass(frozen=True)
class Entity:
    entity_id: EntityId
    canonical_name: str
    aliases: Set[str] = field(default_factory=set)
    gender: GenderEnum = GenderEnum.UNKNOWN
    evidence: Tuple[Evidence, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Mention:
    mention_id: MentionId
    text: str
    start: int
    end: int
    gender: GenderEnum = GenderEnum.UNKNOWN
    entity_id: Optional[EntityId] = None
    evidence: Tuple[Evidence, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StepPlan:
    index: int
    instruction: str
    needed_facts: List[str] = field(default_factory=list)
    depends_on: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class StepDecision:
    action: str
    think: str = ""
    tool_name: str = ""
    tool_input: str = ""
    message: str = ""


@dataclass(frozen=True)
class ToolError:
    kind: str
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class ToolResponse:
    text: str = ""
    raw: Optional[JSONValue] = None


@dataclass(frozen=True)
class StepToolCall:
    tool_call_id: ToolCallId
    tool: str
    tool_input: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    response: Optional[ToolResponse] = None
    error: Optional[ToolError] = None


@dataclass(frozen=True)
class StepMeta:
    action: Optional[str] = None
    tool: Optional[str] = None
    refined_input: Optional[str] = None
    hit: Optional[bool] = None


@dataclass(frozen=True)
class StepResult:
    success: bool
    observation: str = ""
    confidence: float = 0.0

    extracted_facts: Tuple[EvidenceExtractedFact, ...] = field(default_factory=tuple)
    extracted_entities: Tuple[EntityRef, ...] = field(default_factory=tuple)

    error: Optional[str] = None
    meta: StepMeta = field(default_factory=StepMeta)

    should_stop: bool = False
    final_answer: Optional[str] = None


@dataclass(frozen=True)
class Step:
    id: StepId
    plan: StepPlan
    status: StepStatus = StepStatus.PLANNED

    decision: Optional[StepDecision] = None
    tool_calls: Tuple[StepToolCall, ...] = field(default_factory=tuple)
    evidence: Tuple[Evidence, ...] = field(default_factory=tuple)
    result: Optional[StepResult] = None

    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    meta: StepMeta = field(default_factory=StepMeta)


@dataclass(frozen=True)
class Query:
    id: QueryId
    original_text: str
    status: QueryStatus = QueryStatus.PLANNED

    steps: Tuple[Step, ...] = field(default_factory=tuple)
    entities: Tuple[Entity, ...] = field(default_factory=tuple)
    mentions: Tuple[Mention, ...] = field(default_factory=tuple)

    final_answer: Optional[str] = None

    @property
    def entity_index(self) -> Dict[str, Entity]:
        return {str(e.entity_id): e for e in self.entities}


def entity_with_evidence(ent: Entity, ev: Evidence) -> Entity:
    return replace(ent, evidence=ent.evidence + (ev,))


def query_upsert_entity(q: Query, ent: Entity) -> Query:
    ents = list(q.entities)
    for i, e in enumerate(ents):
        if e.entity_id == ent.entity_id:
            ents[i] = ent
            return replace(q, entities=tuple(ents))
    ents.append(ent)
    return replace(q, entities=tuple(ents))


def query_add_mention(q: Query, m: Mention) -> Query:
    return replace(q, mentions=q.mentions + (m,))


def query_add_step(q: Query, step: Step) -> Query:
    return replace(q, steps=q.steps + (step,))


def query_replace_step(q: Query, step: Step) -> Query:
    steps = list(q.steps)
    for i, s in enumerate(steps):
        if s.id == step.id:
            steps[i] = step
            return replace(q, steps=tuple(steps))
    steps.append(step)
    return replace(q, steps=tuple(steps))


def step_add_evidence(step: Step, ev: Evidence) -> Step:
    return replace(step, evidence=step.evidence + (ev,))


def step_add_tool_call(step: Step, tc: StepToolCall) -> Step:
    return replace(step, tool_calls=step.tool_calls + (tc,))


def step_set_decision(step: Step, decision: StepDecision) -> Step:
    return replace(step, decision=decision)


def step_set_result(step: Step, result: StepResult) -> Step:
    return replace(step, result=result)

def new_query(original_text: str) -> Query:
    return Query(
        id=QueryId.generate(),
        original_text=original_text,
    )


def new_step(plan: StepPlan) -> Step:
    return Step(
        id=StepId.generate(),
        plan=plan,
        status=StepStatus.PLANNED,
        started_at=datetime.utcnow(),
    )


def new_entity(
    canonical_name: str,
    *,
    aliases: Optional[Set[str]] = None,
    gender: GenderEnum = GenderEnum.UNKNOWN,
) -> Entity:
    return Entity(
        entity_id=EntityId.generate(),
        canonical_name=canonical_name,
        aliases=aliases or set(),
        gender=gender,
    )


def new_mention(
    text: str,
    start: int,
    end: int,
    *,
    gender: GenderEnum = GenderEnum.UNKNOWN,
    entity_id: Optional[EntityId] = None,
) -> Mention:
    return Mention(
        mention_id=MentionId.generate(),
        text=text,
        start=start,
        end=end,
        gender=gender,
        entity_id=entity_id,
    )


def new_tool_call(tool: str, tool_input: str) -> StepToolCall:
    return StepToolCall(
        tool_call_id=ToolCallId.generate(),
        tool=tool,
        tool_input=tool_input,
        started_at=datetime.utcnow(),
    )
