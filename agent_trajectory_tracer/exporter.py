from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .schemas import EventRecord, ObservationRecord, ScoreRecord, TraceRecord


def make_json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [make_json_safe(v) for v in value]
        return repr(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(make_json_safe(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_safe(row), ensure_ascii=False))
            f.write("\n")


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return cleaned[:80] or "trajectory"


def sort_observations(observations: list[ObservationRecord]) -> list[ObservationRecord]:
    return sorted(
        observations,
        key=lambda obs: (
            obs.start_time,
            obs.end_time or obs.start_time,
            obs.parent_observation_id or "",
            obs.id,
        ),
    )


class LocalTrajectoryExporter:
    def __init__(self, output_root: str | os.PathLike[str] = "output") -> None:
        self.output_root = Path(output_root)

    def export(
        self,
        trace: TraceRecord,
        observations: list[ObservationRecord],
        scores: list[ScoreRecord],
        events: list[EventRecord],
        llm_responses: list[dict[str, Any]] | None = None,
    ) -> Path:
        trace_name = slugify(trace.name or "trajectory")
        timestamp = trace.timestamp.strftime("%Y%m%dT%H%M%S")
        run_dir = self.output_root / f"{timestamp}_{trace_name}_{trace.id[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)

        trace_dict = trace.to_dict()
        sorted_observations = sort_observations(observations)
        observation_dicts = [obs.to_dict() for obs in sorted_observations]
        score_dicts = [score.to_dict() for score in scores]
        event_dicts = [event.to_dict() for event in events]
        llm_response_dicts = llm_responses or []

        write_json(run_dir / "trace.json", trace_dict)
        write_json(run_dir / "trajectory.json", {
            "trace": trace_dict,
            "observations": observation_dicts,
            "scores": score_dicts,
            "events": event_dicts,
        })
        write_json(run_dir / "llm_responses.json", llm_response_dicts)
        write_jsonl(run_dir / "observations.jsonl", observation_dicts)
        write_jsonl(run_dir / "scores.jsonl", score_dicts)
        write_jsonl(run_dir / "events.jsonl", event_dicts)
        write_jsonl(run_dir / "llm_responses.jsonl", llm_response_dicts)
        write_json(run_dir / "summary.json", self._summary(trace, sorted_observations, scores, events))
        (run_dir / "trajectory.md").write_text(
            self._markdown(trace, sorted_observations, scores, events),
            encoding="utf-8",
        )
        self._update_latest_symlink(run_dir)
        return run_dir

    def _summary(
        self,
        trace: TraceRecord,
        observations: list[ObservationRecord],
        scores: list[ScoreRecord],
        events: list[EventRecord],
    ) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        total_usage: dict[str, float] = {}
        total_cost: dict[str, float] = {}
        for obs in observations:
            by_type[obs.type.value] = by_type.get(obs.type.value, 0) + 1
            for key, value in obs.usage_details.items():
                total_usage[key] = total_usage.get(key, 0) + value
            for key, value in obs.cost_details.items():
                total_cost[key] = total_cost.get(key, 0) + value

        return {
            "traceId": trace.id,
            "name": trace.name,
            "status": trace.status,
            "statusMessage": trace.status_message,
            "observationCount": len(observations),
            "scoreCount": len(scores),
            "eventCount": len(events),
            "observationsByType": by_type,
            "usageDetails": total_usage,
            "costDetails": total_cost,
        }

    def _markdown(
        self,
        trace: TraceRecord,
        observations: list[ObservationRecord],
        scores: list[ScoreRecord],
        events: list[EventRecord],
    ) -> str:
        lines = [
            f"# {trace.name or 'Agent Trajectory'}",
            "",
            f"- Trace ID: `{trace.id}`",
            f"- Status: `{trace.status}`",
            f"- Started: `{trace.timestamp.isoformat()}`",
            f"- Observations: `{len(observations)}`",
            f"- Scores: `{len(scores)}`",
            f"- Events: `{len(events)}`",
            "",
            "## Observations",
            "",
        ]

        id_to_depth: dict[str, int] = {}
        for obs in observations:
            parent_depth = id_to_depth.get(obs.parent_observation_id or "", -1)
            depth = parent_depth + 1
            id_to_depth[obs.id] = depth
            indent = "  " * depth
            status = "ERROR" if obs.level.value == "ERROR" else obs.level.value
            lines.append(
                f"{indent}- `{obs.type.value}` **{obs.name or obs.id}** "
                f"({status}, id `{obs.id}`)"
            )

        if scores:
            lines.extend(["", "## Scores", ""])
            for score in scores:
                target = score.observation_id or trace.id
                lines.append(
                    f"- `{score.name}` = `{score.value}` "
                    f"({score.data_type.value}, target `{target}`)"
                )

        return "\n".join(lines) + "\n"

    def _update_latest_symlink(self, run_dir: Path) -> None:
        latest = self.output_root / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_dir.resolve(), target_is_directory=True)
        except OSError:
            # Symlinks are a convenience; exporting the run is the important part.
            pass
