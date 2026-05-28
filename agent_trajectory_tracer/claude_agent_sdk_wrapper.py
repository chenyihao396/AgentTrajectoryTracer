from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Optional

from .schemas import ObservationType


def _message_type(message: Any) -> str:
    return message.__class__.__name__


def current_otel_observation_id() -> str | None:
    """Return the active OpenTelemetry span id if the SDK exposes one."""
    try:
        from opentelemetry import trace
    except ImportError:
        return None

    span = trace.get_current_span()
    context = span.get_span_context()
    if not getattr(context, "is_valid", False):
        return None
    span_id = getattr(context, "span_id", None)
    if not isinstance(span_id, int) or span_id == 0:
        return None
    return f"{span_id:016x}"


def extract_stream_thinking(event: dict[str, Any]) -> dict[str, Any] | None:
    """Extract thinking content from Claude SDK partial stream events."""
    delta = event.get("delta") if isinstance(event, dict) else None
    content_block = event.get("content_block") if isinstance(event, dict) else None

    if isinstance(delta, dict):
        delta_type = delta.get("type")
        if delta_type in {"thinking_delta", "signature_delta"}:
            return {
                "type": delta_type,
                "thinking": delta.get("thinking"),
                "signature_delta": delta.get("signature"),
            }

    if isinstance(content_block, dict):
        block_type = content_block.get("type")
        if block_type in {"thinking", "redacted_thinking"}:
            return {
                "type": block_type,
                "thinking": content_block.get("thinking"),
                "signature": content_block.get("signature"),
                "redacted_data": content_block.get("data"),
            }

    return None


def extract_stream_text(event: dict[str, Any]) -> str | None:
    """Extract answer text from Claude SDK partial stream events."""
    delta = event.get("delta") if isinstance(event, dict) else None
    if not isinstance(delta, dict):
        return None
    if delta.get("type") != "text_delta":
        return None
    text = delta.get("text")
    return str(text) if text else None


class ReasoningAccumulator:
    """Merge streaming thinking deltas into complete reasoning records."""

    def __init__(self) -> None:
        self.thinking_parts: list[str] = []
        self.signature_parts: list[str] = []
        self.blocks: list[dict[str, Any]] = []
        self.redacted_blocks: list[dict[str, Any]] = []

    def add_stream_payload(self, payload: dict[str, Any]) -> None:
        payload_type = payload.get("type")
        if payload_type == "thinking_delta":
            thinking = payload.get("thinking")
            if thinking:
                self.thinking_parts.append(str(thinking))
            return
        if payload_type == "signature_delta":
            signature = payload.get("signature_delta")
            if signature:
                self.signature_parts.append(str(signature))
            return
        if payload_type == "thinking":
            thinking = payload.get("thinking")
            signature = payload.get("signature")
            if thinking or signature:
                self.blocks.append(
                    {
                        "type": "thinking",
                        "thinking": thinking or "",
                        "signature": signature or "",
                    }
                )
            return
        if payload_type == "redacted_thinking":
            self.redacted_blocks.append(
                {
                    "type": "redacted_thinking",
                    "redacted_data": payload.get("redacted_data"),
                }
            )

    def add_assistant_block(self, block: Any) -> None:
        thinking = getattr(block, "thinking", None)
        signature = getattr(block, "signature", None)
        redacted_data = getattr(block, "data", None)
        block_type = block.__class__.__name__

        if redacted_data:
            self.redacted_blocks.append(
                {
                    "type": block_type,
                    "redacted_data": redacted_data,
                }
            )
            return
        if thinking or signature:
            self.blocks.append(
                {
                    "type": block_type,
                    "thinking": thinking or "",
                    "signature": signature or "",
                }
            )

    def to_record(self) -> dict[str, Any] | None:
        merged_thinking = "".join(self.thinking_parts)
        merged_signature = "".join(self.signature_parts)
        records: list[dict[str, Any]] = []

        if not self.blocks and (merged_thinking or merged_signature):
            records.append(
                {
                    "type": "thinking",
                    "thinking": merged_thinking,
                    "signature": merged_signature,
                    "source": "stream_delta",
                }
            )
        records.extend(self.blocks)
        records.extend(self.redacted_blocks)

        if not records:
            return None

        return {
            "records": records,
            "thinking": "\n".join(
                record.get("thinking", "")
                for record in records
                if record.get("thinking")
            ),
            "has_redacted_thinking": bool(self.redacted_blocks),
        }


class TextAccumulator:
    """Merge streaming text deltas, preferring final AssistantMessage text."""

    def __init__(self) -> None:
        self.delta_parts: list[str] = []
        self.blocks: list[str] = []

    def add_delta(self, text: str) -> None:
        self.delta_parts.append(text)

    def add_block(self, text: str) -> None:
        self.blocks.append(text)

    def to_texts(self) -> list[str]:
        if self.blocks:
            return self.blocks
        merged = "".join(self.delta_parts).strip()
        return [merged] if merged else []


class StreamAccumulator:
    """Build a compact record from token-level Claude stream events."""

    def __init__(self) -> None:
        self.event_count = 0
        self.event_types: dict[str, int] = {}
        self.text_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self.signature_parts: list[str] = []
        self.message: dict[str, Any] = {}

    def add_event(self, event: dict[str, Any]) -> None:
        self.event_count += 1
        event_type = str(event.get("type") or "unknown")
        self.event_types[event_type] = self.event_types.get(event_type, 0) + 1

        message = event.get("message")
        if isinstance(message, dict):
            for key in ["id", "model", "role", "stop_reason", "stop_sequence", "usage"]:
                value = message.get(key)
                if value is not None:
                    self.message[key] = value

        delta = event.get("delta")
        if not isinstance(delta, dict):
            return

        delta_type = delta.get("type")
        if delta_type == "text_delta" and delta.get("text"):
            self.text_parts.append(str(delta["text"]))
        elif delta_type == "thinking_delta" and delta.get("thinking"):
            self.thinking_parts.append(str(delta["thinking"]))
        elif delta_type == "signature_delta" and delta.get("signature"):
            self.signature_parts.append(str(delta["signature"]))

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event_count": self.event_count,
            "event_types": self.event_types,
            "message": self.message,
            "text": "".join(self.text_parts),
            "thinking": "".join(self.thinking_parts),
        }
        if self.signature_parts:
            record["signature"] = "".join(self.signature_parts)
        return record


@dataclass
class ClaudeAgentQueryResult:
    prompt: str
    assistant_text: list[str] = field(default_factory=list)
    reasoning: Optional[dict[str, Any]] = None
    result: dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None
    usage_details: dict[str, float] = field(default_factory=dict)
    raw_messages: list[Any] = field(default_factory=list)
    stream: dict[str, Any] = field(default_factory=dict)

    @property
    def answer(self) -> str:
        return "\n".join(text for text in self.assistant_text if text).strip()

    def to_observation_output(self) -> dict[str, Any]:
        output = {
            "content": self.answer,
            "role": "assistant",
        }
        reasoning_content = self.reasoning_content
        if reasoning_content:
            output["reasoning_content"] = reasoning_content
        return output

    def to_trace_output(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
        }

    @property
    def reasoning_content(self) -> str:
        if not self.reasoning:
            return ""
        return str(self.reasoning.get("thinking") or "")

    def to_raw_response(self) -> dict[str, Any]:
        assistant_turns = assistant_turns_from_raw_messages(self.raw_messages)
        return {
            "provider": "claude-agent-sdk",
            "model": self.model,
            "messages": self.to_messages(),
            "assistant_turns": assistant_turns,
            "stream": self.stream,
            "raw_messages": self.raw_messages,
            "result": self.result,
            "output": self.to_observation_output(),
        }

    def to_messages(self) -> list[dict[str, Any]]:
        assistant_message = self.to_observation_output()
        return [
            {
                "role": "user",
                "content": self.prompt,
            },
            assistant_message,
        ]


async def trace_claude_agent_query(
    tracer: Any,
    client: Any,
    prompt: str,
    *,
    name: str = "claude.query",
    metadata: Optional[dict[str, Any]] = None,
    update_trace_output: bool = True,
    record_reasoning_event: bool = False,
) -> ClaudeAgentQueryResult:
    """Run one Claude Agent SDK query and record its intermediate trajectory."""
    messages = [{"role": "user", "content": prompt}]
    tracer.ensure_trace(input={"messages": messages})
    reasoning_accumulator = ReasoningAccumulator()
    text_accumulator = TextAccumulator()
    stream_accumulator = StreamAccumulator()
    query_result = ClaudeAgentQueryResult(prompt=prompt)

    with tracer.start_observation(
        as_type=ObservationType.GENERATION,
        name=name,
        input={"messages": messages},
        metadata=metadata,
    ) as query_observation:
        await client.query(prompt)

        async for message in client.receive_response():
            query_result.raw_messages.append(_dump_message(message))
            raw_event = getattr(message, "event", None)
            if isinstance(raw_event, dict):
                stream_accumulator.add_event(raw_event)
                query_result.model = query_result.model or _model_from_stream_event(raw_event)
                _merge_usage_details(query_result.usage_details, _usage_from_stream_event(raw_event))
                stream_thinking = extract_stream_thinking(raw_event)
                if stream_thinking:
                    reasoning_accumulator.add_stream_payload(stream_thinking)
                stream_text = extract_stream_text(raw_event)
                if stream_text:
                    text_accumulator.add_delta(stream_text)

            if _message_type(message) == "AssistantMessage":
                for block in getattr(message, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        text_accumulator.add_block(text)
                    if (
                        getattr(block, "thinking", None) is not None
                        or getattr(block, "signature", None) is not None
                        or getattr(block, "data", None) is not None
                    ):
                        reasoning_accumulator.add_assistant_block(block)

            if _message_type(message) == "ResultMessage":
                query_result.result = {
                    "subtype": getattr(message, "subtype", None),
                    "duration_ms": getattr(message, "duration_ms", None),
                    "duration_api_ms": getattr(message, "duration_api_ms", None),
                    "total_cost_usd": getattr(message, "total_cost_usd", None),
                    "num_turns": getattr(message, "num_turns", None),
                }

        query_result.reasoning = reasoning_accumulator.to_record()
        query_result.assistant_text = text_accumulator.to_texts()
        query_result.stream = stream_accumulator.to_record()
        if record_reasoning_event and query_result.reasoning is not None:
            _record_reasoning_observation(tracer, query_result.reasoning)

        if query_result.model:
            query_observation.record.name = f"llm.{query_result.model}"

        query_observation.update(
            output=query_result.to_observation_output(),
            metadata={
                "provider": "claude-agent-sdk",
                "result": query_result.result,
            },
            model=query_result.model,
            usage_details=query_result.usage_details,
        )
        tracer.record_llm_response(
            observation_id=query_observation.id,
            model=query_result.model,
            response=query_result.to_raw_response(),
        )

    if update_trace_output:
        tracer.update_trace(output=query_result.to_trace_output())

    return query_result


def _record_reasoning_observation(tracer: Any, reasoning: dict[str, Any]) -> None:
    otel_parent_id = current_otel_observation_id()
    with tracer.start_observation(
        as_type=ObservationType.EVENT,
        name="claude.thinking",
        input={"source": "claude-agent-sdk"},
    ) as observation:
        if otel_parent_id is not None:
            observation.record.parent_observation_id = otel_parent_id
            observation.record.metadata["parentSource"] = "opentelemetry-context"
        else:
            observation.record.metadata["parentSource"] = "query-span"
        observation.update(output=reasoning)


def _dump_message(message: Any) -> Any:
    event = getattr(message, "event", None)
    if isinstance(event, dict):
        return {
            "type": _message_type(message),
            "event": _json_safe_value(event),
        }

    dumped = _json_safe_value(message)
    if isinstance(dumped, dict):
        dumped.setdefault("type", _message_type(message))
        return dumped
    return {
        "type": _message_type(message),
        "repr": repr(message),
        "value": dumped,
    }


def assistant_turns_from_raw_messages(raw_messages: list[Any]) -> list[dict[str, Any]]:
    """Collapse partial AssistantMessage records into assistant turns.

    Claude Agent SDK can emit one AssistantMessage for thinking/text and another
    AssistantMessage with the same message_id for the tool_use block. Grouping
    on message_id recovers the rationale that led to each tool call.
    """
    turns_by_message_id: dict[str, dict[str, Any]] = {}
    ordered_turns: list[dict[str, Any]] = []

    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, dict) or raw_message.get("type") != "AssistantMessage":
            continue
        message_id = raw_message.get("message_id") or raw_message.get("uuid") or f"assistant-{index}"
        message_id = str(message_id)
        turn = turns_by_message_id.get(message_id)
        if turn is None:
            turn = {
                "message_id": message_id,
                "model": raw_message.get("model"),
                "session_id": raw_message.get("session_id"),
                "raw_message_indices": [],
                "reasoning_content": "",
                "content": "",
                "tool_calls": [],
                "usage": {},
            }
            turns_by_message_id[message_id] = turn
            ordered_turns.append(turn)

        turn["raw_message_indices"].append(index)
        if raw_message.get("model") and not turn.get("model"):
            turn["model"] = raw_message.get("model")
        if raw_message.get("session_id") and not turn.get("session_id"):
            turn["session_id"] = raw_message.get("session_id")
        if isinstance(raw_message.get("usage"), dict):
            turn["usage"] = raw_message["usage"]

        for block in raw_message.get("content") or []:
            if not isinstance(block, dict):
                continue
            thinking = block.get("thinking")
            if thinking:
                _append_text_field(turn, "reasoning_content", str(thinking))
            text = block.get("text")
            if text:
                _append_text_field(turn, "content", str(text))
            tool_id = block.get("id")
            tool_name = block.get("name")
            if tool_id or tool_name:
                turn["tool_calls"].append(
                    {
                        "id": tool_id,
                        "name": tool_name,
                        "arguments": block.get("input"),
                    }
                )

    return ordered_turns


def _append_text_field(target: dict[str, Any], key: str, text: str) -> None:
    if not target.get(key):
        target[key] = text
    else:
        target[key] = f"{target[key]}\n{text}"


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]

    if hasattr(value, "model_dump"):
        try:
            return _json_safe_value(value.model_dump(exclude_none=True))
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_safe_value(value.dict())
        except Exception:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return _json_safe_value(asdict(value))
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_safe_value(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return repr(value)


def _model_from_stream_event(event: dict[str, Any]) -> str | None:
    message = event.get("message")
    if isinstance(message, dict):
        model = message.get("model")
        return str(model) if model else None
    return None


def _usage_from_stream_event(event: dict[str, Any]) -> dict[str, float]:
    usage: dict[str, float] = {}
    message = event.get("message")
    delta = event.get("delta")
    raw_usage = None
    if isinstance(message, dict):
        raw_usage = message.get("usage")
    if raw_usage is None and isinstance(delta, dict):
        raw_usage = delta.get("usage")
    if not isinstance(raw_usage, dict):
        return usage

    input_tokens = raw_usage.get("input_tokens")
    output_tokens = raw_usage.get("output_tokens")
    if isinstance(input_tokens, (int, float)):
        usage["input"] = float(input_tokens)
        usage["prompt_tokens"] = float(input_tokens)
    if isinstance(output_tokens, (int, float)):
        usage["output"] = float(output_tokens)
        usage["completion_tokens"] = float(output_tokens)
    if "input" in usage and "output" in usage:
        usage["total"] = usage["input"] + usage["output"]
        usage["total_tokens"] = usage["total"]
    return usage


def _merge_usage_details(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        if value:
            target[key] = value
