from __future__ import annotations

import contextvars
import functools
import traceback
import uuid
import atexit
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar

from .exporter import LocalTrajectoryExporter
from .openai_wrapper import OpenAIClientProxy
from .schemas import (
    EventRecord,
    ObservationLevel,
    ObservationRecord,
    ObservationType,
    ScoreDataType,
    ScoreRecord,
    ScoreSource,
    TraceRecord,
    utc_now,
)


F = TypeVar("F", bound=Callable[..., Any])

_current_tracer: contextvars.ContextVar["AgentTrajectoryTracer | None"] = (
    contextvars.ContextVar("current_agent_trajectory_tracer", default=None)
)


def get_current_tracer() -> "AgentTrajectoryTracer | None":
    return _current_tracer.get()


def _snapshot(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _decode_json_if_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class ObservationHandle:
    def __init__(self, record: ObservationRecord) -> None:
        self.record = record

    @property
    def id(self) -> str:
        return self.record.id

    def update(
        self,
        *,
        input: Any = None,
        output: Any = None,
        metadata: Optional[dict[str, Any]] = None,
        level: ObservationLevel | str | None = None,
        status_message: Optional[str] = None,
        model: Optional[str] = None,
        model_parameters: Any = None,
        usage_details: Optional[dict[str, float]] = None,
        cost_details: Optional[dict[str, float]] = None,
        tool_definitions: Optional[dict[str, str]] = None,
        tool_calls: Optional[list[str]] = None,
        tool_call_names: Optional[list[str]] = None,
    ) -> None:
        if input is not None:
            self.record.input = _snapshot(input)
        if output is not None:
            self.record.output = _snapshot(output)
        if metadata:
            self.record.metadata.update(_snapshot(metadata))
        if level is not None:
            self.record.level = ObservationLevel(level)
        if status_message is not None:
            self.record.status_message = status_message
        if model is not None:
            self.record.model = model
        if model_parameters is not None:
            self.record.model_parameters = _snapshot(model_parameters)
        if usage_details:
            self.record.usage_details.update(_snapshot(usage_details))
        if cost_details:
            self.record.cost_details.update(_snapshot(cost_details))
        if tool_definitions is not None:
            self.record.tool_definitions = _snapshot(tool_definitions)
        if tool_calls is not None:
            self.record.tool_calls = _snapshot(tool_calls)
        if tool_call_names is not None:
            self.record.tool_call_names = _snapshot(tool_call_names)


class AgentTrajectoryTracer:
    """A local, file-backed tracer distilled from Langfuse trace/observation ideas."""

    def __init__(
        self,
        *,
        output_root: str | Path = "output",
        environment: str = "default",
        release: Optional[str] = None,
        version: Optional[str] = None,
        trace_name: str = "agent_trajectory",
        trace_tags: Optional[list[str]] = None,
    ) -> None:
        self.exporter = LocalTrajectoryExporter(output_root)
        self.environment = environment
        self.release = release
        self.version = version
        self.default_trace_name = trace_name
        self.default_trace_tags = trace_tags or []
        self.trace: TraceRecord | None = None
        self.observations: list[ObservationRecord] = []
        self.scores: list[ScoreRecord] = []
        self.events: list[EventRecord] = []
        self.llm_responses: list[dict[str, Any]] = []
        self._tools: dict[str, Callable[..., Any]] = {}
        self._observation_stack: list[ObservationRecord] = []
        self.output_dir: Path | None = None
        self.client: Any = None
        self._otel_adapter: Any = None
        self._exported = False
        self._context_token: contextvars.Token[Any] | None = None
        atexit.register(self.flush)

    def OpenAI(
        self,
        *args: Any,
        trace_name: Optional[str] = None,
        trace_tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create an OpenAI-compatible client wrapped with local tracing.

        Usage:
            tracer.OpenAI(api_key="...", base_url="...")
            response = tracer.client.chat.completions.create(...)
        """
        from openai import OpenAI

        if trace_name is not None:
            self.default_trace_name = trace_name
        if trace_tags is not None:
            self.default_trace_tags = trace_tags
        self.client = OpenAIClientProxy(self, OpenAI(*args, **kwargs))
        return self.client

    def wrap_openai_client(self, client: Any) -> Any:
        """Wrap an existing OpenAI-compatible client instance."""
        self.client = OpenAIClientProxy(self, client)
        return self.client

    def ClaudeAgentSDK(
        self,
        *,
        auto_instrument: bool = True,
        service_name: str = "agent-trajectory-tracer",
        trace_name: Optional[str] = None,
        trace_tags: Optional[list[str]] = None,
    ) -> Any:
        """Enable local tracing for Claude Agent SDK via OpenInference spans.

        Usage:
            tracer.ClaudeAgentSDK()
            ... run normal claude-agent-sdk code ...
            tracer.flush()
        """
        from .otel_adapter import ClaudeAgentSDKAdapter

        if trace_name is not None:
            self.default_trace_name = trace_name
        if trace_tags is not None:
            self.default_trace_tags = trace_tags
        self._otel_adapter = ClaudeAgentSDKAdapter(
            self,
            auto_instrument=auto_instrument,
            service_name=service_name,
        ).install()
        return self._otel_adapter

    async def trace_claude_agent_query(
        self,
        client: Any,
        prompt: str,
        *,
        name: str = "claude.query",
        metadata: Optional[dict[str, Any]] = None,
        update_trace_output: bool = True,
        record_reasoning_event: bool = False,
    ) -> Any:
        """Run one Claude Agent SDK query and record its local trajectory."""
        from .claude_agent_sdk_wrapper import trace_claude_agent_query

        return await trace_claude_agent_query(
            self,
            client,
            prompt,
            name=name,
            metadata=metadata,
            update_trace_output=update_trace_output,
            record_reasoning_event=record_reasoning_event,
        )

    def MiniSweAgent(
        self,
        agent: Any,
        *,
        trace_name: Optional[str] = None,
        trace_tags: Optional[list[str]] = None,
    ) -> Any:
        """Wrap a mini-swe-agent agent so model calls and bash actions are traced."""
        from .mini_swe_agent_wrapper import wrap_mini_swe_agent

        if trace_name is not None:
            self.default_trace_name = trace_name
        if trace_tags is not None:
            self.default_trace_tags = trace_tags
        return wrap_mini_swe_agent(self, agent)

    def wrap_mini_swe_model(self, model: Any) -> Any:
        """Wrap a mini-swe-agent model object with GENERATION tracing."""
        from .mini_swe_agent_wrapper import wrap_mini_swe_model

        return wrap_mini_swe_model(self, model)

    def wrap_mini_swe_environment(self, env: Any) -> Any:
        """Wrap a mini-swe-agent environment object with TOOL tracing."""
        from .mini_swe_agent_wrapper import wrap_mini_swe_environment

        return wrap_mini_swe_environment(self, env)

    def trace_mini_swe_agent_run(
        self,
        agent: Any,
        task: str = "",
        *,
        name: str = "mini-swe-agent.run",
        metadata: Optional[dict[str, Any]] = None,
        update_trace_output: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a mini-swe-agent task and export its trajectory locally."""
        from .mini_swe_agent_wrapper import trace_mini_swe_agent_run

        return trace_mini_swe_agent_run(
            self,
            agent,
            task,
            name=name,
            metadata=metadata,
            update_trace_output=update_trace_output,
            **kwargs,
        )

    def ensure_trace(
        self,
        *,
        name: Optional[str] = None,
        input: Any = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> TraceRecord:
        """Create a trace lazily when wrapper code records the first event."""
        if self.trace is not None:
            return self.trace

        self.trace = TraceRecord(
            id=str(uuid.uuid4()),
            name=name or self.default_trace_name,
            timestamp=utc_now(),
            environment=self.environment,
            release=self.release,
            version=self.version,
            input=_snapshot(input),
            metadata=_snapshot(metadata or {}),
            tags=_snapshot(tags or self.default_trace_tags),
            session_id=session_id,
            user_id=user_id,
        )
        self._context_token = _current_tracer.set(self)
        self._exported = False
        return self.trace

    def flush(self, *, output: Any = None) -> Path | None:
        """Write the current trajectory to output once.

        This is called automatically at process exit, but calling it explicitly
        gives you the output directory immediately.
        """
        if self._otel_adapter is not None:
            self._otel_adapter.ingest_into_tracer()
        if self.trace is None or self._exported:
            return self.output_dir
        if output is not None:
            self.trace.output = _snapshot(output)
        if self.trace.status != "ERROR":
            self.trace.status = "OK"
        self.trace.updated_at = utc_now()
        self.output_dir = self.exporter.export(
            self.trace,
            self.observations,
            self.scores,
            self.events,
            self.llm_responses,
        )
        self._exported = True
        if self._context_token is not None:
            _current_tracer.reset(self._context_token)
            self._context_token = None
        return self.output_dir

    @contextmanager
    def start_trace(
        self,
        *,
        name: str,
        input: Any = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Iterator["AgentTrajectoryTracer"]:
        if self.trace is not None:
            raise RuntimeError("This tracer instance already has an active trace")

        self.trace = TraceRecord(
            id=str(uuid.uuid4()),
            name=name,
            timestamp=utc_now(),
            environment=self.environment,
            release=self.release,
            version=self.version,
            input=_snapshot(input),
            metadata=_snapshot(metadata or {}),
            tags=_snapshot(tags or []),
            session_id=session_id,
            user_id=user_id,
        )
        token = _current_tracer.set(self)
        try:
            yield self
            self.trace.status = "OK"
        except Exception as exc:
            self.trace.status = "ERROR"
            self.trace.status_message = f"{type(exc).__name__}: {exc}"
            self.log_event(
                "trace.exception",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        finally:
            self.trace.updated_at = utc_now()
            self.flush()
            _current_tracer.reset(token)

    @contextmanager
    def start_observation(
        self,
        *,
        as_type: ObservationType | str = ObservationType.SPAN,
        name: Optional[str] = None,
        input: Any = None,
        metadata: Optional[dict[str, Any]] = None,
        model: Optional[str] = None,
        model_parameters: Any = None,
    ) -> Iterator[ObservationHandle]:
        if self.trace is None:
            raise RuntimeError("start_trace must be called before start_observation")

        parent = self._observation_stack[-1] if self._observation_stack else None
        record = ObservationRecord(
            id=str(uuid.uuid4()),
            trace_id=self.trace.id,
            type=ObservationType(as_type),
            start_time=utc_now(),
            name=name,
            input=_snapshot(input),
            metadata=_snapshot(metadata or {}),
            parent_observation_id=parent.id if parent else None,
            model=model,
            model_parameters=_snapshot(model_parameters),
        )
        self.observations.append(record)
        self._observation_stack.append(record)
        handle = ObservationHandle(record)
        try:
            yield handle
        except Exception as exc:
            handle.update(
                level=ObservationLevel.ERROR,
                status_message=f"{type(exc).__name__}: {exc}",
                metadata={"exception": traceback.format_exc()},
            )
            raise
        finally:
            record.end_time = utc_now()
            self._observation_stack.pop()

    def update_current_observation(self, **kwargs: Any) -> None:
        if not self._observation_stack:
            raise RuntimeError("No current observation to update")
        ObservationHandle(self._observation_stack[-1]).update(**kwargs)

    def update_trace(
        self,
        *,
        input: Any = None,
        output: Any = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        status_message: Optional[str] = None,
    ) -> None:
        if self.trace is None:
            raise RuntimeError("start_trace must be called before update_trace")
        if input is not None:
            self.trace.input = _snapshot(input)
        if output is not None:
            self.trace.output = _snapshot(output)
        if metadata:
            self.trace.metadata.update(_snapshot(metadata))
        if tags:
            self.trace.tags.extend(_snapshot(tags))
        if status_message is not None:
            self.trace.status_message = status_message
        self.trace.updated_at = utc_now()

    def log_event(self, name: str, payload: Any = None) -> None:
        if self.trace is None:
            raise RuntimeError("start_trace must be called before log_event")
        parent = self._observation_stack[-1] if self._observation_stack else None
        self.events.append(
            EventRecord(
                id=str(uuid.uuid4()),
                trace_id=self.trace.id,
                observation_id=parent.id if parent else None,
                name=name,
                payload=_snapshot(payload),
            )
        )

    def record_llm_response(
        self,
        *,
        observation_id: str,
        model: Any = None,
        response: Any,
    ) -> None:
        if self.trace is None:
            raise RuntimeError("start_trace must be called before record_llm_response")
        self.llm_responses.append(
            {
                "traceId": self.trace.id,
                "observationId": observation_id,
                "model": model,
                "timestamp": utc_now().isoformat().replace("+00:00", "Z"),
                "response": _snapshot(response),
            }
        )

    def score(
        self,
        *,
        name: str,
        value: Any,
        data_type: ScoreDataType | str = ScoreDataType.NUMERIC,
        source: ScoreSource | str = ScoreSource.API,
        observation_id: Optional[str] = None,
        comment: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ScoreRecord:
        if self.trace is None:
            raise RuntimeError("start_trace must be called before score")
        if observation_id is None and self._observation_stack:
            observation_id = self._observation_stack[-1].id
        record = ScoreRecord(
            id=str(uuid.uuid4()),
            trace_id=self.trace.id,
            observation_id=observation_id,
            name=name,
            value=value,
            data_type=ScoreDataType(data_type),
            source=ScoreSource(source),
            comment=comment,
            metadata=_snapshot(metadata or {}),
        )
        self.scores.append(record)
        return record

    def tool(
        self,
        func: Optional[F] = None,
        *,
        name: Optional[str] = None,
    ) -> F | Callable[[F], F]:
        """Register and wrap a Python tool function with TOOL observation tracing."""

        def decorator(tool_func: F) -> F:
            tool_name = name or tool_func.__name__
            self._tools[tool_name] = tool_func

            @functools.wraps(tool_func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                self.ensure_trace()
                tool_input = kwargs if kwargs else {"args": args}
                with self.start_observation(
                    as_type=ObservationType.TOOL,
                    name=f"tool.{tool_name}",
                    input=tool_input,
                ) as obs:
                    result = tool_func(*args, **kwargs)
                    obs.update(output=_decode_json_if_string(result))
                    return result

            self._tools[tool_name] = wrapper
            return wrapper  # type: ignore[return-value]

        if func is not None:
            return decorator(func)
        return decorator

    def execute_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """Execute a registered OpenAI-style tool call and return a tool message."""
        function = getattr(tool_call, "function", None)
        tool_name = getattr(function, "name", None)
        if not tool_name:
            raise ValueError("tool_call.function.name is required")
        if tool_name not in self._tools:
            raise KeyError(f"Tool is not registered: {tool_name}")

        arguments_raw = getattr(function, "arguments", None) or "{}"
        arguments = json.loads(arguments_raw)
        result = self._tools[tool_name](**arguments)
        content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

        return {
            "role": "tool",
            "tool_call_id": getattr(tool_call, "id", ""),
            "name": tool_name,
            "content": content,
        }

    def observation(
        self,
        *,
        as_type: ObservationType | str = ObservationType.SPAN,
        name: Optional[str] = None,
    ) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.start_observation(
                    as_type=as_type,
                    name=name or func.__name__,
                    input={"args": args, "kwargs": kwargs},
                ) as obs:
                    result = func(*args, **kwargs)
                    obs.update(output=result)
                    return result

            return wrapper  # type: ignore[return-value]

        return decorator

    def trace_function(
        self,
        *,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.start_trace(
                    name=name or func.__name__,
                    input={"args": args, "kwargs": kwargs},
                    tags=tags,
                ):
                    result = func(*args, **kwargs)
                    self.update_trace(output=result)
                    return result

            return wrapper  # type: ignore[return-value]

        return decorator
