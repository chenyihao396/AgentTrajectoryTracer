from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from .schemas import ObservationLevel, ObservationRecord, ObservationType, isoformat, utc_now


OPENINFERENCE_KIND_TO_OBSERVATION = {
    "LLM": ObservationType.GENERATION,
    "AGENT": ObservationType.AGENT,
    "TOOL": ObservationType.TOOL,
    "CHAIN": ObservationType.CHAIN,
    "RETRIEVER": ObservationType.RETRIEVER,
    "EVALUATOR": ObservationType.EVALUATOR,
    "EMBEDDING": ObservationType.EMBEDDING,
    "GUARDRAIL": ObservationType.GUARDRAIL,
}

GENAI_OPERATION_TO_OBSERVATION = {
    "chat": ObservationType.GENERATION,
    "completion": ObservationType.GENERATION,
    "generate_content": ObservationType.GENERATION,
    "execute_tool": ObservationType.TOOL,
    "invoke_agent": ObservationType.AGENT,
    "create_agent": ObservationType.AGENT,
}


def _decode_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _first_present(attributes: Mapping[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in attributes and attributes[key] is not None:
            return _decode_maybe_json(attributes[key])
    return None


def _span_id(value: Any) -> str:
    if value is None:
        return str(uuid.uuid4())
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return f"{value:016x}"
    return str(value)


def _trace_id(value: Any) -> str:
    if value is None:
        return str(uuid.uuid4())
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return f"{value:032x}"
    return str(value)


def _nanos_to_datetime(value: Any) -> datetime:
    if isinstance(value, int):
        return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc)
    if isinstance(value, float):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    return utc_now()


def _normalize_numeric_dicts(value: Any) -> Any:
    if isinstance(value, dict):
        if value and all(str(key).isdigit() for key in value):
            return [
                _normalize_numeric_dicts(value[key])
                for key in sorted(value, key=lambda item: int(str(item)))
            ]
        return {key: _normalize_numeric_dicts(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_numeric_dicts(item) for item in value]
    return _decode_maybe_json(value)


def _nested_attributes(attributes: Mapping[str, Any], prefix: str) -> Any:
    root: dict[str, Any] = {}
    prefix_dot = f"{prefix}."
    for key, value in attributes.items():
        if not key.startswith(prefix_dot):
            continue
        cursor = root
        parts = key[len(prefix_dot) :].split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return _normalize_numeric_dicts(root) if root else None


def _unwrap_openinference_messages(value: Any) -> Any:
    if isinstance(value, list):
        unwrapped = []
        changed = False
        for item in value:
            if isinstance(item, dict) and set(item.keys()) == {"message"}:
                unwrapped.append(item["message"])
                changed = True
            else:
                unwrapped.append(item)
        return unwrapped if changed else value
    return value


def _attributes_from_span(span: Any) -> dict[str, Any]:
    raw = getattr(span, "attributes", None) or {}
    return dict(raw)


def _resource_attributes_from_span(span: Any) -> dict[str, Any]:
    resource = getattr(span, "resource", None)
    raw = getattr(resource, "attributes", None) if resource is not None else None
    return dict(raw or {})


def _scope_from_span(span: Any) -> dict[str, Any]:
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_info", None
    )
    if scope is None:
        return {}
    return {
        "name": getattr(scope, "name", None),
        "version": getattr(scope, "version", None),
    }


def _context_from_span(span: Any) -> tuple[str, str]:
    context = getattr(span, "context", None)
    return _trace_id(getattr(context, "trace_id", None)), _span_id(
        getattr(context, "span_id", None)
    )


def _parent_id_from_span(span: Any) -> Optional[str]:
    parent = getattr(span, "parent", None)
    if parent is None:
        return None
    span_id = getattr(parent, "span_id", None)
    return _span_id(span_id) if span_id is not None else None


def _status_from_span(span: Any) -> tuple[ObservationLevel, Optional[str]]:
    status = getattr(span, "status", None)
    if status is None:
        return ObservationLevel.DEFAULT, None
    code = getattr(status, "status_code", None)
    code_name = str(getattr(code, "name", code) or "").upper()
    description = getattr(status, "description", None)
    if "ERROR" in code_name:
        return ObservationLevel.ERROR, description
    return ObservationLevel.DEFAULT, description


class InMemoryTrajectorySpanExporter:
    """OpenTelemetry exporter that buffers spans for local trajectory export."""

    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans: list[Any]) -> Any:
        self.spans.extend(spans)
        try:
            from opentelemetry.sdk.trace.export import SpanExportResult

            return SpanExportResult.SUCCESS
        except Exception:
            return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class ClaudeAgentSDKAdapter:
    """Bridge Claude Agent SDK OpenInference spans into AgentTrajectoryTracer."""

    def __init__(
        self,
        tracer: Any,
        *,
        auto_instrument: bool = True,
        service_name: str = "agent-trajectory-tracer",
    ) -> None:
        self.tracer = tracer
        self.auto_instrument = auto_instrument
        self.service_name = service_name
        self.exporter = InMemoryTrajectorySpanExporter()
        self._installed = False
        self._ingested_span_ids: set[str] = set()
        self._parent_observation_overrides: dict[str, str] = {}
        self._otel_provider: Any = None

    def install(self) -> "ClaudeAgentSDKAdapter":
        """Install an in-memory OTEL exporter and optionally instrument Claude SDK."""
        if self._installed:
            return self

        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        except ImportError as exc:
            raise ImportError(
                "Claude Agent SDK tracing requires optional dependencies. "
                'Install them with: pip install -e ".[claude]"'
            ) from exc

        processor = SimpleSpanProcessor(self.exporter)
        resource = Resource.create({"service.name": self.service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(processor)
        self._otel_provider = provider

        try:
            otel_trace.set_tracer_provider(provider)
        except Exception:
            existing_provider = otel_trace.get_tracer_provider()
            if hasattr(existing_provider, "add_span_processor"):
                existing_provider.add_span_processor(processor)
                self._otel_provider = existing_provider
            else:
                raise

        if self.auto_instrument:
            self._instrument_claude_agent_sdk()

        self._installed = True
        return self

    def _instrument_claude_agent_sdk(self) -> None:
        try:
            from openinference.instrumentation.claude_agent_sdk import (
                ClaudeAgentSDKInstrumentor,
            )
        except ImportError as exc:
            raise ImportError(
                "openinference-instrumentation-claude-agent-sdk is required. "
                'Install it with: pip install -e ".[claude]"'
            ) from exc

        instrumentor = ClaudeAgentSDKInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=self._otel_provider)
        except TypeError:
            instrumentor.instrument()

    def ingest_into_tracer(self) -> None:
        """Convert buffered OTEL spans into local trace and observation records."""
        if self._otel_provider is not None and hasattr(self._otel_provider, "force_flush"):
            self._otel_provider.force_flush()

        for span in sorted(
            self.exporter.spans,
            key=lambda item: getattr(item, "start_time", 0) or 0,
        ):
            _, span_id = _context_from_span(span)
            if span_id in self._ingested_span_ids:
                continue
            self._ingest_span(span)
            self._ingested_span_ids.add(span_id)

    def _ingest_span(self, span: Any) -> None:
        attributes = _attributes_from_span(span)
        otel_trace_id, otel_span_id = _context_from_span(span)

        trace = self.tracer.ensure_trace(
            name=self._trace_name(span, attributes),
            input=self._trace_input(attributes),
            metadata={
                "source": "claude-agent-sdk",
                "otelTraceId": otel_trace_id,
            },
            tags=self._trace_tags(attributes),
            session_id=attributes.get("session.id") or attributes.get("session_id"),
            user_id=attributes.get("user.id") or attributes.get("user_id"),
        )

        parent_observation_id = _parent_id_from_span(span)
        if parent_observation_id in self._parent_observation_overrides:
            parent_observation_id = self._parent_observation_overrides[parent_observation_id]

        observation = ObservationRecord(
            id=otel_span_id,
            trace_id=trace.id,
            type=self._observation_type(attributes),
            start_time=_nanos_to_datetime(getattr(span, "start_time", None)),
            end_time=_nanos_to_datetime(getattr(span, "end_time", None)),
            name=self._observation_name(span, attributes),
            input=self._observation_input(attributes),
            output=self._observation_output(attributes),
            metadata=self._metadata(span, attributes, otel_trace_id),
            parent_observation_id=parent_observation_id,
            model=self._model(attributes),
            model_parameters=self._model_parameters(attributes),
            usage_details=self._usage_details(attributes),
        )
        observation.level, observation.status_message = _status_from_span(span)

        if self._should_fold_agent_span(observation):
            generation = self._matching_claude_generation(observation)
            if generation is not None:
                self._parent_observation_overrides[observation.id] = generation.id
                self._merge_agent_span_into_generation(generation, observation)
                return

        self.tracer.observations.append(observation)

        trace_output = self._trace_output(attributes)
        if trace.output is None and trace_output is not None:
            trace.output = trace_output
        if observation.type == ObservationType.GENERATION and observation.output is not None:
            self.tracer.record_llm_response(
                observation_id=observation.id,
                model=observation.model,
                response=observation.output,
            )

    def _should_fold_agent_span(self, observation: ObservationRecord) -> bool:
        return (
            observation.type == ObservationType.AGENT
            and observation.name == "ClaudeAgentSDK.ClaudeSDKClient.receive_response"
        )

    def _matching_claude_generation(
        self,
        agent_observation: ObservationRecord,
    ) -> ObservationRecord | None:
        candidates = [
            observation
            for observation in self.tracer.observations
            if observation.type == ObservationType.GENERATION
            and observation.metadata.get("provider") == "claude-agent-sdk"
        ]
        if not candidates:
            return None

        midpoint = agent_observation.start_time
        if agent_observation.end_time is not None:
            midpoint = agent_observation.start_time + (
                agent_observation.end_time - agent_observation.start_time
            ) / 2

        containing = [
            observation
            for observation in candidates
            if observation.start_time <= midpoint
            and (observation.end_time is None or observation.end_time >= midpoint)
        ]
        if containing:
            return min(
                containing,
                key=lambda observation: (
                    observation.end_time or midpoint
                )
                - observation.start_time,
            )

        return min(
            candidates,
            key=lambda observation: abs(
                (observation.start_time - agent_observation.start_time).total_seconds()
            ),
        )

    def _merge_agent_span_into_generation(
        self,
        generation: ObservationRecord,
        agent_observation: ObservationRecord,
    ) -> None:
        generation.metadata.setdefault("openinferenceAgentSpans", []).append(
            {
                "id": agent_observation.id,
                "name": agent_observation.name,
                "input": agent_observation.input,
                "output": agent_observation.output,
                "model": agent_observation.model,
                "startTime": isoformat(agent_observation.start_time),
                "endTime": isoformat(agent_observation.end_time),
                "latency": (
                    (agent_observation.end_time - agent_observation.start_time).total_seconds()
                    if agent_observation.end_time is not None
                    else None
                ),
                "usageDetails": agent_observation.usage_details,
                "metadata": agent_observation.metadata,
                "level": agent_observation.level.value,
                "statusMessage": agent_observation.status_message,
            }
        )

    def _trace_name(self, span: Any, attributes: Mapping[str, Any]) -> Optional[str]:
        return (
            attributes.get("langfuse.trace.name")
            or attributes.get("trace.name")
            or self.tracer.default_trace_name
            or getattr(span, "name", None)
        )

    def _trace_tags(self, attributes: Mapping[str, Any]) -> list[str]:
        tags = attributes.get("langfuse.trace.tags") or attributes.get("trace.tags")
        if isinstance(tags, str):
            decoded = _decode_maybe_json(tags)
            if isinstance(decoded, list):
                return [str(tag) for tag in decoded]
            return [tag.strip() for tag in tags.split(",") if tag.strip()]
        if isinstance(tags, (list, tuple)):
            return [str(tag) for tag in tags]
        return list(self.tracer.default_trace_tags)

    def _observation_type(self, attributes: Mapping[str, Any]) -> ObservationType:
        kind = attributes.get("openinference.span.kind")
        if isinstance(kind, str) and kind.upper() in OPENINFERENCE_KIND_TO_OBSERVATION:
            return OPENINFERENCE_KIND_TO_OBSERVATION[kind.upper()]
        operation = attributes.get("gen_ai.operation.name")
        if isinstance(operation, str) and operation in GENAI_OPERATION_TO_OBSERVATION:
            return GENAI_OPERATION_TO_OBSERVATION[operation]
        if self._model(attributes):
            return ObservationType.GENERATION
        return ObservationType.SPAN

    def _observation_name(self, span: Any, attributes: Mapping[str, Any]) -> str:
        return str(
            attributes.get("langfuse.observation.name")
            or attributes.get("gen_ai.tool.name")
            or attributes.get("tool.name")
            or getattr(span, "name", None)
            or "span"
        )

    def _trace_input(self, attributes: Mapping[str, Any]) -> Any:
        explicit = _first_present(attributes, ["langfuse.trace.input", "trace.input"])
        if explicit is not None:
            return explicit
        return self._observation_input(attributes)

    def _trace_output(self, attributes: Mapping[str, Any]) -> Any:
        explicit = _first_present(attributes, ["langfuse.trace.output", "trace.output"])
        if explicit is not None:
            return explicit
        return self._observation_output(attributes)

    def _observation_input(self, attributes: Mapping[str, Any]) -> Any:
        explicit = _first_present(
            attributes,
            [
                "langfuse.observation.input",
                "input.value",
                "gen_ai.prompt",
                "gen_ai.input.messages",
                "llm.prompts",
                "gen_ai.tool.call.arguments",
                "tool.parameters",
            ],
        )
        if explicit is not None:
            return explicit

        messages = _nested_attributes(attributes, "llm.input_messages")
        if messages is not None:
            return {"messages": _unwrap_openinference_messages(messages)}
        return None

    def _observation_output(self, attributes: Mapping[str, Any]) -> Any:
        explicit = _first_present(
            attributes,
            [
                "langfuse.observation.output",
                "output.value",
                "gen_ai.completion",
                "gen_ai.output.messages",
                "gen_ai.tool.call.result",
                "tool.output",
            ],
        )
        if explicit is not None:
            return explicit

        messages = _nested_attributes(attributes, "llm.output_messages")
        if messages is not None:
            return {"messages": _unwrap_openinference_messages(messages)}
        return None

    def _model(self, attributes: Mapping[str, Any]) -> Optional[str]:
        value = _first_present(
            attributes,
            [
                "llm.model_name",
                "gen_ai.request.model",
                "gen_ai.response.model",
                "model",
            ],
        )
        return str(value) if value is not None else None

    def _model_parameters(self, attributes: Mapping[str, Any]) -> Any:
        parameters = _first_present(attributes, ["llm.invocation_parameters"])
        if isinstance(parameters, dict):
            return parameters
        result = {}
        for key in [
            "gen_ai.request.temperature",
            "gen_ai.request.top_p",
            "gen_ai.request.max_tokens",
            "gen_ai.request.frequency_penalty",
            "gen_ai.request.presence_penalty",
        ]:
            if key in attributes:
                result[key.rsplit(".", 1)[-1]] = attributes[key]
        return result or parameters

    def _usage_details(self, attributes: Mapping[str, Any]) -> dict[str, float]:
        usage: dict[str, float] = {}
        mappings = {
            "llm.token_count.prompt": ["input", "prompt_tokens"],
            "llm.token_count.completion": ["output", "completion_tokens"],
            "llm.token_count.total": ["total", "total_tokens"],
            "gen_ai.usage.input_tokens": ["input", "prompt_tokens"],
            "gen_ai.usage.output_tokens": ["output", "completion_tokens"],
            "gen_ai.usage.total_tokens": ["total", "total_tokens"],
        }
        for attr_key, usage_keys in mappings.items():
            value = attributes.get(attr_key)
            if isinstance(value, (int, float)):
                for usage_key in usage_keys:
                    usage[usage_key] = float(value)
        if "total" not in usage and "input" in usage and "output" in usage:
            usage["total"] = usage["input"] + usage["output"]
            usage["total_tokens"] = usage["total"]
        return usage

    def _metadata(
        self,
        span: Any,
        attributes: Mapping[str, Any],
        otel_trace_id: str,
    ) -> dict[str, Any]:
        metadata = {
            "source": "claude-agent-sdk",
            "otelTraceId": otel_trace_id,
            "resource": _resource_attributes_from_span(span),
            "instrumentationScope": _scope_from_span(span),
            "attributes": dict(attributes),
        }
        events = []
        for event in getattr(span, "events", []) or []:
            events.append(
                {
                    "name": getattr(event, "name", None),
                    "timestamp": _nanos_to_datetime(getattr(event, "timestamp", None))
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "attributes": dict(getattr(event, "attributes", None) or {}),
                }
            )
        if events:
            metadata["spanEvents"] = events
        return metadata
