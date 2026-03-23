"""MCP tools for scheduling lighting transitions."""

from datetime import datetime
from typing import Any

from claude_agent_sdk import tool

from fiat_lux.scheduler import (
    cancel_scheduled,
    list_scheduled,
    schedule_transition,
)


def _text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _error(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


@tool(
    "schedule_transition",
    "Schedule a gradual lighting transition at a specific time. The lights will "
    "be set to start_state at start_time, then gradually ramp to end_state over "
    "duration_minutes. Uses the Hue bridge's native transition for smooth fading. "
    "Great for sunrise alarms, gradual wake-ups, or timed wind-downs. "
    "Max ramp duration is ~109 minutes (Hue bridge limit).",
    {
        "type": "object",
        "properties": {
            "start_time": {
                "type": "string",
                "description": (
                    "When to begin the transition, ISO 8601 format "
                    "(e.g. '2026-03-24T08:00:00'). Must be in the future."
                ),
            },
            "lights": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Light names to control. Use ['all'] for all lights.",
            },
            "start_state": {
                "type": "object",
                "description": (
                    "Initial lighting state at start_time. Keys: "
                    "brightness_pct (0-100), kelvin (2000-6500)."
                ),
                "properties": {
                    "brightness_pct": {"type": "number"},
                    "kelvin": {"type": "number"},
                },
            },
            "end_state": {
                "type": "object",
                "description": (
                    "Final lighting state after the ramp. Keys: "
                    "brightness_pct (0-100), kelvin (2000-6500)."
                ),
                "properties": {
                    "brightness_pct": {"type": "number"},
                    "kelvin": {"type": "number"},
                },
            },
            "duration_minutes": {
                "type": "number",
                "description": "How long the gradual transition takes, in minutes. Max ~109.",
            },
            "description": {
                "type": "string",
                "description": "Human-readable description (e.g. 'Sunrise wake-up ramp').",
            },
        },
        "required": [
            "start_time",
            "lights",
            "start_state",
            "end_state",
            "duration_minutes",
        ],
    },
)
async def schedule_transition_tool(args: dict[str, Any]) -> dict[str, Any]:
    try:
        start_time = datetime.fromisoformat(args["start_time"])
    except ValueError:
        return _error(
            f"Invalid time format: {args['start_time']}. "
            "Use ISO 8601 (e.g. '2026-03-24T08:00:00')."
        )

    if start_time <= datetime.now():
        return _error("Start time must be in the future.")

    duration = args["duration_minutes"]
    if duration <= 0 or duration > 109:
        return _error("Duration must be between 0 and 109 minutes.")

    job_id = schedule_transition(
        start_time=start_time,
        lights=args["lights"],
        start_state=args["start_state"],
        end_state=args["end_state"],
        duration_minutes=duration,
        description=args.get("description", ""),
    )

    time_str = start_time.strftime("%H:%M")
    end_str = f"{int(duration)} min"
    return _text(
        f"Scheduled (id={job_id}): {args.get('description', 'Transition')} "
        f"at {time_str}, ramping over {end_str}.\n"
        f"The daemon will execute this automatically — no need to keep the CLI open."
    )


@tool(
    "list_scheduled",
    "List all pending scheduled lighting transitions. Shows job ID, time, "
    "description, and lights affected.",
    {},
)
async def list_scheduled_tool(args: dict[str, Any]) -> dict[str, Any]:
    jobs = list_scheduled()
    if not jobs:
        return _text("No scheduled transitions pending.")

    lines = ["**Scheduled transitions:**\n"]
    for job in jobs:
        try:
            start = datetime.fromisoformat(job["start_time"])
            time_str = start.strftime("%a %H:%M")
        except ValueError:
            time_str = job.get("start_time", "?")

        desc = job.get("description", "Transition")
        duration = job.get("duration_minutes", "?")
        lights = ", ".join(job.get("lights", []))
        lines.append(
            f"  - **{desc}** (id={job['id']}): "
            f"{time_str}, {duration}min ramp, lights: {lights}"
        )

    return _text("\n".join(lines))


@tool(
    "cancel_scheduled",
    "Cancel a pending scheduled lighting transition by its job ID.",
    {"job_id": str},
)
async def cancel_scheduled_tool(args: dict[str, Any]) -> dict[str, Any]:
    job_id = args["job_id"]
    if cancel_scheduled(job_id):
        return _text(f"Cancelled scheduled job {job_id}.")
    return _error(f"No pending job found with id '{job_id}'.")


ALL_SCHEDULER_TOOLS = [
    schedule_transition_tool,
    list_scheduled_tool,
    cancel_scheduled_tool,
]
