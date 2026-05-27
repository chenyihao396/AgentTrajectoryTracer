from __future__ import annotations

import copy
import traceback
from dataclasses import dataclass
from typing import Any

from .schemas import ObservationLevel, ObservationType


def _snapshot(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _model_name(model: Any) -> str:
    config = getattr(model, "config", None)
    if config is not None:
        name = getattr(config, "model_name", None)
        if name:
            return str(name)
    return model.__class__.__name__


def _model_parameters(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")
    if hasattr(config, "dict"):
        return config.dict()
    return {}


def _usage_from_response(response: Any) -> dict[str, float]:
    if not isinstance(response, dict):
        return {}
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        return {}

    mapping = {
        "prompt_tokens": ("input", "prompt_tokens"),
        "input_tokens": ("input", "prompt_tokens"),
        "completion_tokens": ("output", "completion_tokens"),
        "output_tokens": ("output", "completion_tokens"),
        "total_tokens": ("total", "total_tokens"),
    }
    details: dict[str, float] = {}
    for source_key, target_keys in mapping.items():
        value = usage.get(source_key)
        if isinstance(value, (int, float)):
            for target_key in target_keys:
                details[target_key] = float(value)
    return details


def _action_ids(actions: list[dict[str, Any]]) -> list[str]:
    ids = []
    for action in actions:
        if action_id := action.get("tool_call_id"):
            ids.append(str(action_id))
    return ids


def _compact_message(message: dict[str, Any]) -> dict[str, Any]:
    compact = {k: _snapshot(v) for k, v in message.items() if k != "extra"}
    extra = message.get("extra") or {}
    if isinstance(extra, dict):
        if actions := extra.get("actions"):
            compact["actions"] = _snapshot(actions)
        if "cost" in extra:
            compact["cost"] = extra.get("cost")
        if "timestamp" in extra:
            compact["timestamp"] = extra.get("timestamp")
    return compact


def _env_metadata(env: Any) -> dict[str, Any]:
    config = getattr(env, "config", None)
    if config is None:
        return {"environmentType": f"{env.__class__.__module__}.{env.__class__.__name__}"}
    if hasattr(config, "model_dump"):
        config_dict = config.model_dump(mode="json")
    elif hasattr(config, "dict"):
        config_dict = config.dict()
    else:
        config_dict = {}
    return {
        "environmentType": f"{env.__class__.__module__}.{env.__class__.__name__}",
        "environmentConfig": config_dict,
    }


def _command_summary(action: dict[str, Any], limit: int = 80) -> str:
    command = str(action.get("command", "")).strip().splitlines()[0] if action.get("command") else ""
    if not command:
        return "tool.bash"
    return f"tool.bash {command[:limit]}"


@dataclass
class MiniSweAgentTraceState:
    last_generation_id: str | None = None
    step_index: int = 0


class TracedMiniSweModel:
    """Duck-typed wrapper for mini-swe-agent model objects."""

    _agent_trajectory_tracer_wrapped = True

    def __init__(self, tracer: Any, model: Any, state: MiniSweAgentTraceState | None = None) -> None:
        self._tracer = tracer
        self._model = model
        self._state = state or MiniSweAgentTraceState()

    @property
    def trace_state(self) -> MiniSweAgentTraceState:
        return self._state

    @property
    def wrapped_model(self) -> Any:
        return self._model

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        tracer = self._tracer
        tracer.ensure_trace(
            input={"messages": _snapshot(messages)},
            metadata={"source": "mini-swe-agent"},
        )
        name = _model_name(self._model)
        self._state.step_index += 1

        with tracer.start_observation(
            as_type=ObservationType.GENERATION,
            name=f"llm.{name}",
            input={"messages": _snapshot(messages)},
            metadata={
                "source": "mini-swe-agent",
                "step": self._state.step_index,
                "modelType": f"{self._model.__class__.__module__}.{self._model.__class__.__name__}",
            },
            model=name,
            model_parameters=_model_parameters(self._model) | {"query_kwargs": _snapshot(kwargs)},
        ) as obs:
            message = self._model.query(messages, **kwargs)
            extra = message.get("extra") or {}
            response = extra.get("response")
            actions = extra.get("actions") or []

            usage_details = _usage_from_response(response)
            cost_details = {}
            if isinstance(extra.get("cost"), (int, float)):
                cost_details["total"] = float(extra["cost"])

            obs.update(
                output=_compact_message(message),
                usage_details=usage_details,
                cost_details=cost_details,
                tool_calls=_action_ids(actions) or None,
                tool_call_names=["bash"] * len(actions) if actions else None,
            )
            tracer.record_llm_response(
                observation_id=obs.id,
                model=name,
                response=response if response is not None else _compact_message(message),
            )
            self._state.last_generation_id = obs.id
            return message

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class TracedMiniSweEnvironment:
    """Duck-typed wrapper for mini-swe-agent environment objects."""

    _agent_trajectory_tracer_wrapped = True

    def __init__(self, tracer: Any, env: Any, state: MiniSweAgentTraceState | None = None) -> None:
        self._tracer = tracer
        self._env = env
        self._state = state or MiniSweAgentTraceState()

    @property
    def trace_state(self) -> MiniSweAgentTraceState:
        return self._state

    @property
    def wrapped_environment(self) -> Any:
        return self._env

    def execute(self, action: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        tracer = self._tracer
        tracer.ensure_trace(metadata={"source": "mini-swe-agent"})

        pending_exception: Exception | None = None
        output: dict[str, Any] | None = None
        with tracer.start_observation(
            as_type=ObservationType.TOOL,
            name=_command_summary(action),
            input=_snapshot(action),
            metadata={
                "source": "mini-swe-agent",
                "tool": "bash",
                "toolCallId": action.get("tool_call_id"),
                **_env_metadata(self._env),
            },
        ) as obs:
            if self._state.last_generation_id:
                obs.record.parent_observation_id = self._state.last_generation_id
            try:
                output = self._env.execute(action, *args, **kwargs)
            except Exception as exc:
                interrupt_messages = getattr(exc, "messages", None)
                level = ObservationLevel.DEFAULT if type(exc).__name__ == "Submitted" else ObservationLevel.ERROR
                obs.update(
                    output={
                        "exception": str(exc),
                        "exceptionType": type(exc).__name__,
                        "messages": _snapshot(interrupt_messages),
                    },
                    level=level,
                    status_message=f"{type(exc).__name__}: {exc}",
                    metadata=(
                        {}
                        if type(exc).__name__ == "Submitted"
                        else {"exceptionTraceback": traceback.format_exc()}
                    ),
                )
                pending_exception = exc
            else:
                level = ObservationLevel.DEFAULT
                status_message = None
                if output.get("exception_info"):
                    level = ObservationLevel.ERROR
                    status_message = str(output.get("exception_info"))
                obs.update(output=_snapshot(output), level=level, status_message=status_message)

        if pending_exception is not None:
            raise pending_exception
        return output or {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def _shared_state(tracer: Any) -> MiniSweAgentTraceState:
    state = getattr(tracer, "_mini_swe_trace_state", None)
    if state is None:
        state = MiniSweAgentTraceState()
        setattr(tracer, "_mini_swe_trace_state", state)
    return state


def wrap_mini_swe_model(tracer: Any, model: Any, state: MiniSweAgentTraceState | None = None) -> TracedMiniSweModel:
    if getattr(model, "_agent_trajectory_tracer_wrapped", False):
        return model
    return TracedMiniSweModel(tracer, model, state or _shared_state(tracer))


def wrap_mini_swe_environment(
    tracer: Any,
    env: Any,
    state: MiniSweAgentTraceState | None = None,
) -> TracedMiniSweEnvironment:
    if getattr(env, "_agent_trajectory_tracer_wrapped", False):
        return env
    return TracedMiniSweEnvironment(tracer, env, state or _shared_state(tracer))


def wrap_mini_swe_agent(tracer: Any, agent: Any) -> Any:
    state = MiniSweAgentTraceState()
    setattr(tracer, "_mini_swe_trace_state", state)
    agent.model = wrap_mini_swe_model(tracer, agent.model, state)
    agent.env = wrap_mini_swe_environment(tracer, agent.env, state)
    return agent


def trace_mini_swe_agent_run(
    tracer: Any,
    agent: Any,
    task: str = "",
    *,
    name: str = "mini-swe-agent.run",
    metadata: dict[str, Any] | None = None,
    update_trace_output: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    agent = wrap_mini_swe_agent(tracer, agent)
    tracer.ensure_trace(
        name=tracer.default_trace_name,
        input={"task": task, "kwargs": _snapshot(kwargs)},
        metadata={"source": "mini-swe-agent", **(metadata or {})},
    )
    with tracer.start_observation(
        as_type=ObservationType.AGENT,
        name=name,
        input={"task": task, "kwargs": _snapshot(kwargs)},
        metadata={
            "source": "mini-swe-agent",
            "agentType": f"{agent.__class__.__module__}.{agent.__class__.__name__}",
        },
    ) as obs:
        result = agent.run(task, **kwargs)
        obs.update(output=_snapshot(result))
        if update_trace_output:
            tracer.update_trace(output=_snapshot(result))
        return result
