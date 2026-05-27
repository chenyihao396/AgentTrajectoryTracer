from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agent_trajectory_tracer import AgentTrajectoryTracer


PROJECT_ROOT = Path(__file__).resolve().parent


def validate_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    has_anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_custom_provider = bool(os.getenv("ANTHROPIC_BASE_URL")) and bool(
        os.getenv("ANTHROPIC_AUTH_TOKEN")
    )
    if not has_anthropic_key and not has_custom_provider:
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY, or set ANTHROPIC_BASE_URL and "
            "ANTHROPIC_AUTH_TOKEN for an Anthropic-compatible provider."
        )


async def main() -> None:
    validate_env()

    tracer = AgentTrajectoryTracer(
        output_root=PROJECT_ROOT / "output",
        trace_name="claude_agent_sdk_demo",
        trace_tags=["claude-agent-sdk", "deepseek-compatible"],
    )
    tracer.ClaudeAgentSDK()

    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a concise security assistant. Answer briefly and do not "
            "modify files."
        ),
        allowed_tools=["Read", "Glob"],
        permission_mode="acceptEdits",
        cwd=str(PROJECT_ROOT),
        max_turns=5,
        include_partial_messages=True,
        thinking={"type": "enabled", "budget_tokens": 2048},
        effort="max",
    )

    prompt = (
        "Read README.md in the current directory, then explain in one "
        "short paragraph what this project records in an agent trajectory."
    )

    async with ClaudeSDKClient(options=options) as client:
        result = await tracer.trace_claude_agent_query(client, prompt)

    output_dir = tracer.flush()
    print(
        json.dumps(
            {
                "answer": result.answer,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
