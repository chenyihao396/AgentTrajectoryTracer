from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from .schemas import ObservationLevel, ObservationRecord, ObservationType, isoformat, utc_now

_CLAUDE_AGENT_SDK_INSTRUMENTED = False


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


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        return str(text) if text is not None else json.dumps(content, ensure_ascii=False)
    return str(content)


def _tool_calls_from_message(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls is None:
        return []
    if isinstance(raw_tool_calls, dict):
        raw_tool_calls = [raw_tool_calls]
    if not isinstance(raw_tool_calls, list):
        return []

    tool_calls = []
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue
        tool_call = raw_tool_call.get("tool_call")
        if not isinstance(tool_call, dict):
            tool_call = raw_tool_call

        function = tool_call.get("function")
        if not isinstance(function, dict):
            function = {}
        arguments = function.get("arguments", tool_call.get("arguments"))
        tool_calls.append(
            {
                "id": tool_call.get("id"),
                "name": function.get("name") or tool_call.get("name"),
                "arguments": _decode_maybe_json(arguments),
            }
        )

    return [
        tool_call
        for tool_call in tool_calls
        if tool_call.get("id") or tool_call.get("name") or tool_call.get("arguments") is not None
    ]


def _claude_output_messages_from_attributes(
    attributes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_messages = _nested_attributes(attributes, "llm.output_messages")
    if raw_messages is None:
        return []
    if isinstance(raw_messages, dict):
        raw_messages = [raw_messages]
    if not isinstance(raw_messages, list):
        return []

    messages = []
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, dict):
            continue
        message = raw_message.get("message")
        if not isinstance(message, dict):
            message = raw_message
        content = _content_to_text(message.get("content")).strip()
        messages.append(
            {
                "index": index,
                "role": message.get("role"),
                "content": content,
                "tool_calls": _tool_calls_from_message(message),
            }
        )
    return messages


def _tool_id_from_attributes(attributes: Mapping[str, Any]) -> str | None:
    tool_id = _first_present(
        attributes,
        [
            "tool.id",
            "gen_ai.tool.call.id",
            "gen_ai.tool_call.id",
        ],
    )
    return str(tool_id) if tool_id is not None else None


def _index_assistant_turn_contexts(assistant_turns: list[Any]) -> dict[str, Any]:
    by_tool_id: dict[str, dict[str, Any]] = {}
    content_turns: list[dict[str, Any]] = []
    for turn in assistant_turns:
        if not isinstance(turn, dict):
            continue
        if turn.get("content"):
            content_turns.append(turn)
        for tool_call in turn.get("tool_calls") or []:
            if not isinstance(tool_call, dict) or not tool_call.get("id"):
                continue
            by_tool_id[str(tool_call["id"])] = turn
    return {
        "by_tool_id": by_tool_id,
        "content_turns": content_turns,
    }


def _turn_context_for_message(
    message: Mapping[str, Any],
    turn_contexts: Mapping[str, Any],
) -> dict[str, Any]:
    by_tool_id = turn_contexts.get("by_tool_id")
    if isinstance(by_tool_id, dict):
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict) or not tool_call.get("id"):
                continue
            turn = by_tool_id.get(str(tool_call["id"]))
            if isinstance(turn, dict):
                return turn

    content = message.get("content")
    if content:
        content_turns = turn_contexts.get("content_turns")
        if isinstance(content_turns, list):
            for turn in content_turns:
                if not isinstance(turn, dict):
                    continue
                turn_content = str(turn.get("content") or "")
                if turn_content and (
                    turn_content == content
                    or turn_content in str(content)
                    or str(content) in turn_content
                ):
                    return turn
    return {}


def _usage_details_from_turn_context(turn_context: Mapping[str, Any]) -> dict[str, float]:
    usage = turn_context.get("usage")
    if not isinstance(usage, dict):
        return {}

    result: dict[str, float] = {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cache_creation = usage.get("cache_creation_input_tokens")
    cache_read = usage.get("cache_read_input_tokens")
    if isinstance(input_tokens, (int, float)):
        result["input"] = float(input_tokens)
        result["prompt_tokens"] = float(input_tokens)
    if isinstance(output_tokens, (int, float)):
        result["output"] = float(output_tokens)
        result["completion_tokens"] = float(output_tokens)
    if isinstance(cache_creation, (int, float)):
        result["cache_creation_input_tokens"] = float(cache_creation)
    if isinstance(cache_read, (int, float)):
        result["cache_read_input_tokens"] = float(cache_read)
    if "input" in result and "output" in result:
        result["total"] = result["input"] + result["output"]
        result["total_tokens"] = result["total"]
    return result


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
        self._tool_parent_observation_overrides: dict[str, str] = {}
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
        existing_provider = otel_trace.get_tracer_provider()
        if hasattr(existing_provider, "add_span_processor"):
            existing_provider.add_span_processor(processor)
            self._otel_provider = existing_provider
        else:
            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(processor)
            otel_trace.set_tracer_provider(provider)
            self._otel_provider = provider

        if self.auto_instrument:
            self._instrument_claude_agent_sdk()

        self._installed = True
        return self

    def _instrument_claude_agent_sdk(self) -> None:
        global _CLAUDE_AGENT_SDK_INSTRUMENTED
        if _CLAUDE_AGENT_SDK_INSTRUMENTED:
            return

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
        _CLAUDE_AGENT_SDK_INSTRUMENTED = True

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
        tool_parent_observation_id = self._tool_parent_override(attributes)
        if tool_parent_observation_id is not None:
            parent_observation_id = tool_parent_observation_id
        elif parent_observation_id in self._parent_observation_overrides:
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
        self._link_existing_assistant_messages_after_tool(observation)

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
        tool_parent_overrides = self._materialize_claude_output_messages(
            generation,
            agent_observation,
        )
        if tool_parent_overrides:
            self._tool_parent_observation_overrides.update(tool_parent_overrides)
            self._retarget_existing_tool_observations(tool_parent_overrides)

    def _materialize_claude_output_messages(
        self,
        generation: ObservationRecord,
        agent_observation: ObservationRecord,
    ) -> dict[str, str]:
        attributes = (
            agent_observation.metadata.get("attributes", {})
            if isinstance(agent_observation.metadata, dict)
            else {}
        )
        if not isinstance(attributes, dict):
            return {}

        messages = _claude_output_messages_from_attributes(attributes)
        if not messages:
            return {}

        turn_contexts = self._assistant_turn_contexts(generation.id)
        tool_observation_ids = self._tool_observation_ids_by_tool_id()
        tool_parent_overrides: dict[str, str] = {}
        previous_observation_id: str | None = generation.id
        previous_turn_id: str | None = None
        previous_turn_tool_call_ids: list[str] = []
        created_count = 0

        for message in messages:
            content = message.get("content")
            tool_calls = message.get("tool_calls") or []
            if not content and not tool_calls:
                continue

            message_index = int(message["index"])
            observation_id = f"{agent_observation.id}-message-{message_index}"
            if any(observation.id == observation_id for observation in self.tracer.observations):
                continue

            turn_context = _turn_context_for_message(message, turn_contexts)
            turn_id = (
                str(turn_context.get("message_id"))
                if isinstance(turn_context.get("message_id"), str)
                else None
            )
            after_tool_call_ids: list[str] = []
            if turn_id is not None and turn_id != previous_turn_id:
                after_tool_call_ids = list(previous_turn_tool_call_ids)

            after_tool_observation_ids = [
                tool_observation_ids[tool_call_id]
                for tool_call_id in after_tool_call_ids
                if tool_call_id in tool_observation_ids
            ]
            parent_observation_id = (
                after_tool_observation_ids[-1]
                if after_tool_observation_ids
                else previous_observation_id or generation.id
            )

            timestamp = agent_observation.start_time + timedelta(microseconds=created_count)
            if agent_observation.end_time is not None and timestamp > agent_observation.end_time:
                timestamp = agent_observation.end_time
            end_time = timestamp

            input_payload = self._assistant_message_input(
                generation,
                parent_observation_id,
                after_tool_call_ids,
                after_tool_observation_ids,
            )
            output = {
                "role": message.get("role") or "assistant",
                "message_index": message_index,
            }
            reasoning_content = turn_context.get("reasoning_content")
            if reasoning_content:
                output["reasoning_content"] = reasoning_content
            output["content"] = content or turn_context.get("content") or ""
            if tool_calls:
                output["tool_calls"] = tool_calls
            if turn_context.get("message_id"):
                output["message_id"] = turn_context["message_id"]

            observation = ObservationRecord(
                id=observation_id,
                trace_id=generation.trace_id,
                type=ObservationType.GENERATION,
                start_time=timestamp,
                end_time=end_time,
                name=f"claude.assistant_message.{message_index}",
                input=input_payload,
                output=output,
                metadata={
                    "provider": "claude-agent-sdk",
                    "source": "openinference.llm.output_messages",
                    "openinferenceAgentSpanId": agent_observation.id,
                    "messageIndex": message_index,
                    "rootGenerationObservationId": generation.id,
                },
                parent_observation_id=parent_observation_id,
                model=generation.model,
                usage_details=_usage_details_from_turn_context(turn_context),
            )
            if turn_context.get("raw_message_indices"):
                observation.metadata["rawMessageIndices"] = turn_context["raw_message_indices"]

            self.tracer.observations.append(observation)
            created_count += 1

            for tool_call in tool_calls:
                tool_id = tool_call.get("id")
                if tool_id:
                    tool_parent_overrides[str(tool_id)] = observation.id

            current_tool_call_ids = [
                str(tool_call["id"])
                for tool_call in tool_calls
                if tool_call.get("id")
            ]
            previous_observation_id = observation.id
            if turn_id is not None and turn_id != previous_turn_id:
                previous_turn_id = turn_id
                previous_turn_tool_call_ids = current_tool_call_ids
            elif current_tool_call_ids:
                previous_turn_tool_call_ids.extend(current_tool_call_ids)

        return tool_parent_overrides

    def _assistant_message_input(
        self,
        generation: ObservationRecord,
        parent_observation_id: str,
        after_tool_call_ids: list[str],
        after_tool_observation_ids: list[str],
    ) -> dict[str, Any]:
        if after_tool_call_ids or after_tool_observation_ids:
            payload: dict[str, Any] = {
                "previousObservationId": parent_observation_id,
            }
            if after_tool_call_ids:
                payload["afterToolCallIds"] = after_tool_call_ids
            if after_tool_observation_ids:
                payload["afterToolObservationIds"] = after_tool_observation_ids
            return payload

        if parent_observation_id != generation.id:
            return {
                "previousObservationId": parent_observation_id,
            }

        return {
            "messages": (
                generation.input.get("messages")
                if isinstance(generation.input, dict) and "messages" in generation.input
                else generation.input
            )
        }

    def _assistant_turn_contexts(self, generation_id: str) -> dict[str, Any]:
        for llm_response in getattr(self.tracer, "llm_responses", []) or []:
            if llm_response.get("observationId") != generation_id:
                continue
            response = llm_response.get("response")
            if not isinstance(response, dict):
                continue
            assistant_turns = response.get("assistant_turns")
            if assistant_turns is None and isinstance(response.get("raw_messages"), list):
                from .claude_agent_sdk_wrapper import assistant_turns_from_raw_messages

                assistant_turns = assistant_turns_from_raw_messages(response["raw_messages"])
            if isinstance(assistant_turns, list):
                return _index_assistant_turn_contexts(assistant_turns)
        return {"by_tool_id": {}, "content_turns": []}

    def _tool_observation_ids_by_tool_id(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for observation in self.tracer.observations:
            if observation.type != ObservationType.TOOL:
                continue
            metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
            attributes = metadata.get("attributes", {})
            if not isinstance(attributes, dict):
                continue
            tool_id = _tool_id_from_attributes(attributes)
            if tool_id is not None:
                result[tool_id] = observation.id
        return result

    def _tool_parent_override(self, attributes: Mapping[str, Any]) -> str | None:
        tool_id = _tool_id_from_attributes(attributes)
        if tool_id is None:
            return None
        return self._tool_parent_observation_overrides.get(tool_id)

    def _retarget_existing_tool_observations(
        self,
        tool_parent_overrides: dict[str, str],
    ) -> None:
        for observation in self.tracer.observations:
            if observation.type != ObservationType.TOOL:
                continue
            metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
            attributes = metadata.get("attributes", {})
            if not isinstance(attributes, dict):
                continue
            tool_id = _tool_id_from_attributes(attributes)
            if tool_id is not None and tool_id in tool_parent_overrides:
                observation.parent_observation_id = tool_parent_overrides[tool_id]

    def _link_existing_assistant_messages_after_tool(
        self,
        tool_observation: ObservationRecord,
    ) -> None:
        if tool_observation.type != ObservationType.TOOL:
            return
        metadata = tool_observation.metadata if isinstance(tool_observation.metadata, dict) else {}
        attributes = metadata.get("attributes", {})
        if not isinstance(attributes, dict):
            return
        tool_id = _tool_id_from_attributes(attributes)
        if tool_id is None:
            return

        for observation in self.tracer.observations:
            if observation.type != ObservationType.GENERATION:
                continue
            input_payload = observation.input
            if not isinstance(input_payload, dict):
                continue
            after_tool_call_ids = input_payload.get("afterToolCallIds")
            if not isinstance(after_tool_call_ids, list):
                continue
            if tool_id not in {str(item) for item in after_tool_call_ids}:
                continue

            existing_ids = input_payload.get("afterToolObservationIds")
            if not isinstance(existing_ids, list):
                existing_ids = []
            if tool_observation.id not in existing_ids:
                input_payload["afterToolObservationIds"] = [
                    *existing_ids,
                    tool_observation.id,
                ]
            if observation.parent_observation_id != tool_observation.id:
                observation.metadata.setdefault("parentObservationOverride", {
                    "previousParentObservationId": observation.parent_observation_id,
                    "reason": "afterToolCallIds",
                })
                observation.parent_observation_id = tool_observation.id

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
