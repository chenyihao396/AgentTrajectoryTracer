from __future__ import annotations

from typing import Any

from .schemas import ObservationLevel, ObservationType


class OpenAIClientProxy:
    def __init__(self, tracer: Any, client: Any) -> None:
        self._tracer = tracer
        self._client = client
        self.chat = ChatProxy(tracer, client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class ChatProxy:
    def __init__(self, tracer: Any, chat: Any) -> None:
        self._chat = chat
        self.completions = CompletionsProxy(tracer, chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class CompletionsProxy:
    def __init__(self, tracer: Any, completions: Any) -> None:
        self._tracer = tracer
        self._completions = completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)

    def create(self, **kwargs: Any) -> Any:
        self._tracer.ensure_trace(
            input={"messages": kwargs.get("messages")},
        )

        model = kwargs.get("model")
        observation_input = _request_input(kwargs)
        model_parameters = _model_parameters(kwargs)
        with self._tracer.start_observation(
            as_type=ObservationType.GENERATION,
            name=f"llm.{model or 'chat.completions'}",
            input=observation_input,
            model=model,
            model_parameters=model_parameters,
        ) as observation:
            try:
                response = self._completions.create(**kwargs)
            except Exception as exc:
                observation.update(
                    level=ObservationLevel.ERROR,
                    status_message=f"{type(exc).__name__}: {exc}",
                    metadata={"provider": "openai-compatible"},
                )
                raise

            observation.update(
                output=_response_output(response),
                usage_details=_usage_details(response),
                metadata=_response_metadata(response),
                tool_calls=_tool_call_ids(response),
                tool_call_names=_tool_call_names(response),
            )
            self._tracer.record_llm_response(
                observation_id=observation.id,
                model=model,
                response=_dump(response),
            )
            return response


def _request_input(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": kwargs.get("messages"),
        "tools": kwargs.get("tools"),
        "tool_choice": kwargs.get("tool_choice"),
        "response_format": kwargs.get("response_format"),
    }


def _model_parameters(kwargs: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "messages",
        "tools",
        "tool_choice",
        "response_format",
    }
    return {key: value for key, value in kwargs.items() if key not in excluded}


def _response_output(response: Any) -> Any:
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        if message is not None:
            return _dump(message)
    return _dump(response)


def _response_metadata(response: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"provider": "openai-compatible"}
    for attr in ("id", "object", "created", "model", "system_fingerprint"):
        value = getattr(response, attr, None)
        if value is not None:
            metadata[attr] = value
    return metadata


def _usage_details(response: Any) -> dict[str, float]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    usage_details: dict[str, float] = {}
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if prompt_tokens is not None:
        usage_details["input"] = float(prompt_tokens)
        usage_details["prompt_tokens"] = float(prompt_tokens)
    if completion_tokens is not None:
        usage_details["output"] = float(completion_tokens)
        usage_details["completion_tokens"] = float(completion_tokens)
    if total_tokens is not None:
        usage_details["total"] = float(total_tokens)
        usage_details["total_tokens"] = float(total_tokens)
    return usage_details


def _tool_call_ids(response: Any) -> list[str] | None:
    tool_calls = _tool_calls(response)
    if not tool_calls:
        return None
    return [str(getattr(tool_call, "id", "")) for tool_call in tool_calls]


def _tool_call_names(response: Any) -> list[str] | None:
    tool_calls = _tool_calls(response)
    if not tool_calls:
        return None
    names = []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None)
        if name:
            names.append(str(name))
    return names or None


def _tool_calls(response: Any) -> Any:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    if message is None:
        return None
    return getattr(message, "tool_calls", None)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict()
    return value
