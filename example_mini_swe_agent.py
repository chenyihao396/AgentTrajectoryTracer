from __future__ import annotations

import json
import sys
from pathlib import Path

from agent_trajectory_tracer import AgentTrajectoryTracer


PROJECT_ROOT = Path(__file__).resolve().parent
MINI_SWE_AGENT_ROOT = PROJECT_ROOT.parent / "mini-swe-agent"
if MINI_SWE_AGENT_ROOT.exists():
    sys.path.insert(0, str(MINI_SWE_AGENT_ROOT / "src"))


def main() -> None:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.test_models import DeterministicModel, make_output

    tracer = AgentTrajectoryTracer(
        output_root=PROJECT_ROOT / "output",
        trace_name="mini_swe_agent_demo",
        trace_tags=["mini-swe-agent", "deterministic"],
    )

    model = DeterministicModel(
        outputs=[
            make_output(
                "THOUGHT: I should inspect the current project root.\n\n"
                "```mswea_bash_command\n"
                "pwd && test -f README.md && echo README_FOUND\n"
                "```",
                [{"command": "pwd && test -f README.md && echo README_FOUND"}],
                cost=0.0,
            ),
            make_output(
                "THOUGHT: The check passed, so I can submit a concise result.\n\n"
                "```mswea_bash_command\n"
                "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nREADME.md exists and mini-swe-agent tracing works.\\n'\n"
                "```",
                [
                    {
                        "command": (
                            "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n"
                            "README.md exists and mini-swe-agent tracing works.\\n'"
                        )
                    }
                ],
                cost=0.0,
            ),
        ],
        cost_per_call=0.0,
    )
    env = LocalEnvironment(cwd=str(PROJECT_ROOT), timeout=10)
    agent = DefaultAgent(
        model,
        env,
        system_template="You are a deterministic mini-swe-agent demo.",
        instance_template="Task: {{task}}",
        cost_limit=0.0,
        step_limit=5,
    )

    result = tracer.trace_mini_swe_agent_run(
        agent,
        "Verify this project has a README and submit a one-line result.",
    )
    output_dir = tracer.flush()
    print(json.dumps({"result": result, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
