from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from agent_trajectory_tracer import AgentTrajectoryTracer


load_dotenv()


def main() -> None:
    tracer = AgentTrajectoryTracer(
        output_root="output",
        trace_name="deepseek_weather_agent",
        trace_tags=["openai-wrapper", "deepseek"],
    )

    @tracer.tool
    def get_weather(location: str) -> str:
        return json.dumps(
            {
                "location": location,
                "temperature": "72",
                "unit": "fahrenheit",
                "forecast": ["sunny", "windy"],
            }
        )

    tracer.OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What's the weather like in Boston?"},
    ]

    response = tracer.client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        messages=messages,
        tools=tools,
        stream=False,
        reasoning_effort=os.getenv("DEEPSEEK_REASONING_EFFORT", "high"),
        extra_body={"thinking": {"type": "enabled"}},
    )

    response_message = response.choices[0].message
    messages.append(response_message.model_dump(exclude_none=True))

    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            if function_name != "get_weather":
                continue

            messages.append(tracer.execute_tool_call(tool_call))

        second_response = tracer.client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            messages=messages,
            tools=tools,
            stream=False,
            reasoning_effort=os.getenv("DEEPSEEK_REASONING_EFFORT", "high"),
            extra_body={"thinking": {"type": "enabled"}},
        )
        final_answer = second_response.choices[0].message.content
    else:
        final_answer = response_message.content

    output_dir = tracer.flush(output={"answer": final_answer})
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
