from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


JsonValue = Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat().replace("+00:00", "Z") if dt else None


class ObservationType(str, Enum):
    SPAN = "SPAN"
    EVENT = "EVENT"
    GENERATION = "GENERATION"
    AGENT = "AGENT"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"
    EVALUATOR = "EVALUATOR"
    EMBEDDING = "EMBEDDING"
    GUARDRAIL = "GUARDRAIL"


class ObservationLevel(str, Enum):
    DEBUG = "DEBUG"
    DEFAULT = "DEFAULT"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ScoreSource(str, Enum):
    API = "API"
    EVAL = "EVAL"
    ANNOTATION = "ANNOTATION"


class ScoreDataType(str, Enum):
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    BOOLEAN = "BOOLEAN"
    CORRECTION = "CORRECTION"
    TEXT = "TEXT"


@dataclass
class TraceRecord:
    id: str
    name: Optional[str]
    timestamp: datetime
    environment: str = "default"
    tags: list[str] = field(default_factory=list)
    release: Optional[str] = None
    version: Optional[str] = None
    input: JsonValue = None
    output: JsonValue = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    status: str = "OK"
    status_message: Optional[str] = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "name": self.name,
            "timestamp": isoformat(self.timestamp),
            "environment": self.environment,
            "tags": self.tags,
            "release": self.release,
            "version": self.version,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "sessionId": self.session_id,
            "userId": self.user_id,
            "createdAt": isoformat(self.created_at),
            "updatedAt": isoformat(self.updated_at),
            "status": self.status,
            "statusMessage": self.status_message,
        }


@dataclass
class ObservationRecord:
    id: str
    trace_id: str
    type: ObservationType
    start_time: datetime
    name: Optional[str] = None
    input: JsonValue = None
    output: JsonValue = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    parent_observation_id: Optional[str] = None
    level: ObservationLevel = ObservationLevel.DEFAULT
    status_message: Optional[str] = None
    model: Optional[str] = None
    model_parameters: JsonValue = None
    completion_start_time: Optional[datetime] = None
    usage_details: dict[str, float] = field(default_factory=dict)
    cost_details: dict[str, float] = field(default_factory=dict)
    tool_definitions: Optional[dict[str, str]] = None
    tool_calls: Optional[list[str]] = None
    tool_call_names: Optional[list[str]] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> dict[str, JsonValue]:
        latency = None
        if self.end_time is not None:
            latency = (self.end_time - self.start_time).total_seconds()
        return {
            "id": self.id,
            "traceId": self.trace_id,
            "type": self.type.value,
            "startTime": isoformat(self.start_time),
            "endTime": isoformat(self.end_time),
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "parentObservationId": self.parent_observation_id,
            "level": self.level.value,
            "statusMessage": self.status_message,
            "model": self.model,
            "modelParameters": self.model_parameters,
            "completionStartTime": isoformat(self.completion_start_time),
            "latency": latency,
            "usageDetails": self.usage_details,
            "costDetails": self.cost_details,
            "toolDefinitions": self.tool_definitions,
            "toolCalls": self.tool_calls,
            "toolCallNames": self.tool_call_names,
        }


@dataclass
class ScoreRecord:
    id: str
    trace_id: str
    name: str
    value: JsonValue
    data_type: ScoreDataType
    source: ScoreSource = ScoreSource.API
    observation_id: Optional[str] = None
    comment: Optional[str] = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "traceId": self.trace_id,
            "observationId": self.observation_id,
            "name": self.name,
            "value": self.value,
            "dataType": self.data_type.value,
            "source": self.source.value,
            "comment": self.comment,
            "metadata": self.metadata,
            "timestamp": isoformat(self.timestamp),
        }


@dataclass
class EventRecord:
    id: str
    trace_id: str
    name: str
    payload: JsonValue = None
    observation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "traceId": self.trace_id,
            "observationId": self.observation_id,
            "name": self.name,
            "payload": self.payload,
            "timestamp": isoformat(self.timestamp),
        }
