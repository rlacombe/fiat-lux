"""Lighting scheduler — schedule transitions for future execution.

Jobs are persisted to ~/.config/fiat_lux/schedule.json so they survive
daemon restarts. The daemon runs a background loop that checks for due
jobs every 10 seconds.

The Hue bridge natively supports transition times up to ~109 minutes
(65535 deciseconds). At the scheduled start time, we set the initial
state instantly, then send the end state with a long transitiontime —
the bridge handles the gradual ramp.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fiat_lux.mcp.hue import _get_bridge, _normalize

CONFIG_DIR = Path.home() / ".config" / "fiat_lux"
SCHEDULE_FILE = CONFIG_DIR / "schedule.json"

POLL_INTERVAL = 10  # seconds


def _load_schedule() -> list[dict[str, Any]]:
    if SCHEDULE_FILE.exists():
        try:
            return json.loads(SCHEDULE_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _save_schedule(jobs: list[dict[str, Any]]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SCHEDULE_FILE.write_text(json.dumps(jobs, indent=2))


def schedule_transition(
    start_time: datetime,
    lights: list[str],
    start_state: dict[str, Any],
    end_state: dict[str, Any],
    duration_minutes: float,
    description: str = "",
) -> str:
    """Schedule a gradual lighting transition.

    Returns the job ID.
    """
    job_id = uuid.uuid4().hex[:8]
    job = {
        "id": job_id,
        "start_time": start_time.isoformat(),
        "lights": lights,
        "start_state": start_state,
        "end_state": end_state,
        "duration_minutes": duration_minutes,
        "description": description,
        "created": datetime.now().isoformat(),
    }
    jobs = _load_schedule()
    jobs.append(job)
    _save_schedule(jobs)
    return job_id


def list_scheduled() -> list[dict[str, Any]]:
    """Return all pending scheduled jobs."""
    now = datetime.now()
    jobs = _load_schedule()
    # Only return future jobs
    pending = []
    for job in jobs:
        try:
            start = datetime.fromisoformat(job["start_time"])
            # Include jobs that haven't finished yet (start + duration)
            end = start.timestamp() + job.get("duration_minutes", 0) * 60
            if end > now.timestamp():
                pending.append(job)
        except (ValueError, KeyError):
            continue
    return pending


def cancel_scheduled(job_id: str) -> bool:
    """Cancel a scheduled job by ID. Returns True if found."""
    jobs = _load_schedule()
    original_len = len(jobs)
    jobs = [j for j in jobs if j.get("id") != job_id]
    if len(jobs) < original_len:
        _save_schedule(jobs)
        return True
    return False


def _resolve_lights(bridge, light_names: list[str]) -> list[int]:
    """Resolve light names to IDs."""
    if light_names == ["all"]:
        return [l.light_id for l in bridge.lights]

    name_map = {_normalize(l.name).lower(): l.light_id for l in bridge.lights}
    ids = []
    for name in light_names:
        lid = name_map.get(_normalize(name).lower())
        if lid is not None:
            ids.append(lid)
    return ids


def _interpolate_value(start_val: float, end_val: float, progress: float) -> float:
    """Linearly interpolate between start and end based on progress (0.0–1.0)."""
    return start_val + (end_val - start_val) * progress


def _execute_transition(job: dict[str, Any]) -> None:
    """Execute a scheduled lighting transition.

    If we're exactly on time: set start state, then ramp to end state.
    If we're late (laptop woke up after start): jump to the interpolated
    position and ramp the remaining duration.
    """
    b = _get_bridge()
    light_ids = _resolve_lights(b, job["lights"])
    if not light_ids:
        print(f"  [scheduler] No lights found for job {job['id']}", flush=True)
        return

    start_state = job.get("start_state", {})
    end_state = job.get("end_state", {})
    duration_minutes = job.get("duration_minutes", 1)
    duration_seconds = duration_minutes * 60

    # How late are we?
    start_time = datetime.fromisoformat(job["start_time"])
    elapsed = (datetime.now() - start_time).total_seconds()
    progress = max(0.0, min(1.0, elapsed / duration_seconds))
    remaining_seconds = max(0, duration_seconds - elapsed)

    if remaining_seconds < 5:
        # Almost done — just jump to end state
        end_cmd: dict[str, Any] = {"on": True, "transitiontime": 10}
        if "brightness_pct" in end_state:
            end_cmd["bri"] = max(1, round(end_state["brightness_pct"] * 254 / 100))
        if "kelvin" in end_state:
            end_cmd["ct"] = round(1_000_000 / end_state["kelvin"])
        for lid in light_ids:
            b.set_light(lid, end_cmd)
        print(f"  [scheduler] Caught up (nearly done): {job.get('description', job['id'])}", flush=True)
        return

    # Step 1: Jump to current interpolated position
    now_cmd: dict[str, Any] = {"on": True, "transitiontime": 0}
    if "brightness_pct" in start_state and "brightness_pct" in end_state:
        bri_pct = _interpolate_value(start_state["brightness_pct"], end_state["brightness_pct"], progress)
        now_cmd["bri"] = max(1, round(bri_pct * 254 / 100))
    elif "brightness_pct" in start_state:
        now_cmd["bri"] = max(1, round(start_state["brightness_pct"] * 254 / 100))
    if "kelvin" in start_state and "kelvin" in end_state:
        kelvin = _interpolate_value(start_state["kelvin"], end_state["kelvin"], progress)
        now_cmd["ct"] = round(1_000_000 / kelvin)
    elif "kelvin" in start_state:
        now_cmd["ct"] = round(1_000_000 / start_state["kelvin"])

    for lid in light_ids:
        b.set_light(lid, now_cmd)

    time.sleep(0.3)

    # Step 2: Ramp to end state over remaining duration
    transitiontime = round(remaining_seconds * 10)
    transitiontime = min(transitiontime, 65535)

    end_cmd = {"on": True, "transitiontime": transitiontime}
    if "brightness_pct" in end_state:
        end_cmd["bri"] = max(1, round(end_state["brightness_pct"] * 254 / 100))
    if "kelvin" in end_state:
        end_cmd["ct"] = round(1_000_000 / end_state["kelvin"])

    for lid in light_ids:
        b.set_light(lid, end_cmd)

    desc = job.get("description", job["id"])
    late_str = f" (caught up, {int(elapsed)}s late)" if progress > 0.05 else ""
    print(
        f"  [scheduler] Started: {desc} "
        f"({remaining_seconds / 60:.0f}min remaining on {len(light_ids)} lights){late_str}",
        flush=True,
    )


def _cleanup_past_jobs() -> None:
    """Remove jobs that have fully completed (start + duration is past)."""
    now = datetime.now()
    jobs = _load_schedule()
    active = []
    for job in jobs:
        try:
            start = datetime.fromisoformat(job["start_time"])
            end_ts = start.timestamp() + job.get("duration_minutes", 0) * 60
            if end_ts > now.timestamp():
                active.append(job)
        except (ValueError, KeyError):
            continue
    if len(active) != len(jobs):
        _save_schedule(active)


async def scheduler_loop() -> None:
    """Background loop that checks for due jobs and executes them.

    Runs forever until cancelled. Safe to run as an asyncio.Task.
    Catches up on missed jobs (e.g. after laptop wake from sleep).
    """
    print("  Scheduler: active", flush=True)

    # Track which jobs we've already fired so we don't re-fire them
    fired_ids: set[str] = set()

    while True:
        try:
            now = datetime.now()
            jobs = _load_schedule()

            for job in jobs:
                if job["id"] in fired_ids:
                    continue

                try:
                    start = datetime.fromisoformat(job["start_time"])
                except (ValueError, KeyError):
                    continue

                duration_seconds = job.get("duration_minutes", 0) * 60
                end_time = start.timestamp() + duration_seconds
                seconds_until = (start - now).total_seconds()

                # Fire if: start time has arrived AND transition hasn't ended yet
                if seconds_until <= POLL_INTERVAL and now.timestamp() < end_time:
                    await asyncio.to_thread(_execute_transition, job)
                    fired_ids.add(job["id"])

            # Clean up old jobs periodically
            await asyncio.to_thread(_cleanup_past_jobs)
            # Also clean up fired_ids for removed jobs
            current_ids = {j.get("id") for j in jobs}
            fired_ids &= current_ids

        except Exception as e:
            print(f"  [scheduler] error: {e}", flush=True)

        await asyncio.sleep(POLL_INTERVAL)
