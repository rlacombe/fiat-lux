"""Fiat-Lux daemon — persistent ClaudeSDKClient with Unix socket interface.

Boots once, keeps the Claude session warm. The CLI connects to the socket
to send prompts and stream responses. Shortcuts are intercepted before
hitting the LLM.
"""

import asyncio
import json
import os
import signal
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    create_sdk_mcp_server,
)

from fiat_lux.alerts import alert_loop
from fiat_lux.pulse import breathing_mode_loop, candle_mode_loop
from fiat_lux.routines import ALL_ROUTINE_TOOLS
from fiat_lux.scheduler import scheduler_loop
from fiat_lux.shortcuts import (
    try_shortcut,
    SHORTCUT_BREATHE_START,
    SHORTCUT_BREATHE_STOP,
    SHORTCUT_CANDLE_START,
)
from fiat_lux.tools.calendar_tools import ALL_CALENDAR_TOOLS
from fiat_lux.tools.circadian import get_circadian_recommendation
from fiat_lux.tools.hue import ALL_HUE_TOOLS, get_lights_context
from fiat_lux.tools.memory import ALL_MEMORY_TOOLS, get_profile_context
from fiat_lux.tools.weather_tools import ALL_WEATHER_TOOLS
from fiat_lux.weather import get_weather_context
from fiat_lux.tools.scheduler_tools import ALL_SCHEDULER_TOOLS

SOCKET_PATH = Path.home() / ".config" / "fiat_lux" / "lux.sock"
PID_FILE = Path.home() / ".config" / "fiat_lux" / "lux.pid"
BASE_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text()

# Background breathing mode task
_breathing_task: asyncio.Task | None = None


def _build_system_prompt() -> str:
    from datetime import datetime

    parts = [BASE_SYSTEM_PROMPT]
    parts.append(f"## Current Time\n{datetime.now().strftime('%A %Y-%m-%d %H:%M')}")
    profile = get_profile_context()
    if profile:
        parts.append(profile)
    lights = get_lights_context()
    if lights:
        parts.append(lights)
    weather = get_weather_context()
    if weather:
        parts.append(weather)
    return "\n\n".join(parts)


def _refresh_dynamic_prompt(options: ClaudeAgentOptions) -> None:
    """Update time and weather in the system prompt."""
    from datetime import datetime
    import re

    prompt = options.system_prompt

    # Refresh time
    new_time = f"## Current Time\n{datetime.now().strftime('%A %Y-%m-%d %H:%M')}"
    prompt = re.sub(r"## Current Time\n.+", new_time, prompt)

    # Refresh weather (replace existing section or append)
    weather = get_weather_context()
    if "## Current Weather" in prompt:
        if weather:
            prompt = re.sub(
                r"## Current Weather\n(?:- .+\n?)+",
                weather + "\n",
                prompt,
            )
        else:
            prompt = re.sub(r"\n\n## Current Weather\n(?:- .+\n?)+", "", prompt)
    elif weather:
        # Append weather after the time section
        prompt = prompt.replace(new_time, f"{new_time}\n\n{weather}")

    options.system_prompt = prompt


def _build_options() -> ClaudeAgentOptions:
    all_sdk_tools = [
        get_circadian_recommendation,
        *ALL_HUE_TOOLS,
        *ALL_MEMORY_TOOLS,
        *ALL_ROUTINE_TOOLS,
        *ALL_CALENDAR_TOOLS,
        *ALL_SCHEDULER_TOOLS,
        *ALL_WEATHER_TOOLS,
    ]
    fiat_lux_tools = create_sdk_mcp_server(
        name="fiat_lux",
        version="0.1.0",
        tools=all_sdk_tools,
    )
    # Pre-load tool names so Claude doesn't need ToolSearch
    tool_names = [f"mcp__fiat_lux__{t.name}" for t in all_sdk_tools]
    return ClaudeAgentOptions(
        system_prompt=_build_system_prompt(),
        mcp_servers={"fiat_lux": fiat_lux_tools},
        tools=tool_names,
        allowed_tools=["mcp__fiat_lux__*"],
        permission_mode="acceptEdits",
        max_turns=10,
        setting_sources=[],
    )


async def _stop_breathing() -> bool:
    """Stop the breathing mode if active. Returns True if it was running."""
    global _breathing_task
    if _breathing_task is not None and not _breathing_task.done():
        _breathing_task.cancel()
        try:
            await _breathing_task
        except asyncio.CancelledError:
            pass
        _breathing_task = None
        return True
    _breathing_task = None
    return False


def _resolve_light_ids(light_name: str) -> list[int] | None:
    """Resolve a light name to IDs. Returns None for all lights."""
    if not light_name:
        return None
    try:
        from fiat_lux.tools.hue import _get_bridge, _normalize
        b = _get_bridge()
        name_map = {_normalize(l.name).lower(): l.light_id for l in b.lights}
        lid = name_map.get(_normalize(light_name).lower())
        if lid is not None:
            return [lid]
        # Try partial match
        for lname, lid in name_map.items():
            if light_name.lower() in lname:
                return [lid]
    except Exception:
        pass
    return None


async def _handle_ambient(shortcut_result: str) -> str:
    """Handle ambient mode start/stop signals from shortcuts."""
    global _breathing_task

    # Parse optional light name and fade duration from sentinel
    # Format: SENTINEL or SENTINEL:light_name or SENTINEL:light_name:fade_minutes
    light_name = ""
    fade_minutes = 0.0
    if ":" in shortcut_result:
        parts = shortcut_result.split(":", 2)
        shortcut_result = parts[0]
        light_name = parts[1] if len(parts) > 1 else ""
        if len(parts) > 2 and parts[2]:
            try:
                fade_minutes = float(parts[2])
            except ValueError:
                pass
    light_ids = _resolve_light_ids(light_name)
    light_label = f" on {light_name}" if light_name else ""
    fade_label = f", fading out over {int(fade_minutes)}min" if fade_minutes else ""

    if shortcut_result == SHORTCUT_BREATHE_START:
        await _stop_breathing()
        _breathing_task = asyncio.create_task(breathing_mode_loop(light_ids))
        return f"Breathing mode started{light_label}. Say 'stop' to end."

    if shortcut_result == SHORTCUT_CANDLE_START:
        await _stop_breathing()
        _breathing_task = asyncio.create_task(candle_mode_loop(light_ids, fade_out_minutes=fade_minutes))
        return f"Candle mode started{light_label}{fade_label}. Say 'stop' to end."

    if shortcut_result == SHORTCUT_BREATHE_STOP:
        was_running = await _stop_breathing()
        if was_running:
            return "Ambient mode stopped. Lights restored."
        # Not in ambient mode — just turn everything off
        from fiat_lux.shortcuts import _all_off
        return _all_off()

    # Any other shortcut: stop ambient mode if active
    if _breathing_task is not None and not _breathing_task.done():
        await _stop_breathing()

    return shortcut_result


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    client: ClaudeSDKClient,
    options: ClaudeAgentOptions,
):
    """Handle a single CLI connection."""
    try:
        # Refresh the time in the system prompt so it's always current
        _refresh_dynamic_prompt(options)

        data = await reader.readline()
        if not data:
            return

        request = json.loads(data.decode())
        prompt = request.get("prompt", "")

        if not prompt:
            writer.write(json.dumps({"type": "done"}).encode() + b"\n")
            await writer.drain()
            return

        # Tier 1: shortcuts — instant, no LLM
        shortcut_result = try_shortcut(prompt)
        if shortcut_result is not None:
            # Handle breathing mode start/stop
            shortcut_result = await _handle_ambient(shortcut_result)

            writer.write(
                json.dumps({"type": "text", "text": shortcut_result}).encode() + b"\n"
            )
            writer.write(json.dumps({"type": "done"}).encode() + b"\n")
            await writer.drain()
            return

        # Any non-shortcut command stops breathing mode
        await _stop_breathing()

        # Tier 2: Claude via persistent ClaudeSDKClient
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text") and block.text:
                        writer.write(
                            json.dumps({"type": "text", "text": block.text}).encode()
                            + b"\n"
                        )
                        await writer.drain()
                    elif hasattr(block, "name"):
                        writer.write(
                            json.dumps(
                                {"type": "tool", "name": block.name}
                            ).encode()
                            + b"\n"
                        )
                        await writer.drain()
            elif isinstance(message, ResultMessage):
                if message.subtype != "success":
                    writer.write(
                        json.dumps(
                            {"type": "error", "text": message.subtype}
                        ).encode()
                        + b"\n"
                    )
                    await writer.drain()

        writer.write(json.dumps({"type": "done"}).encode() + b"\n")
        await writer.drain()

    except Exception as e:
        try:
            writer.write(
                json.dumps({"type": "error", "text": str(e)}).encode() + b"\n"
            )
            writer.write(json.dumps({"type": "done"}).encode() + b"\n")
            await writer.drain()
        except Exception:
            pass
    finally:
        writer.close()
        await writer.wait_closed()


async def run_daemon() -> None:
    """Start the Fiat-Lux daemon."""
    # Clean up stale socket
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

    # Write PID file
    PID_FILE.write_text(str(os.getpid()))

    options = _build_options()

    print("Lux daemon starting...", flush=True)
    print(f"  Socket: {SOCKET_PATH}", flush=True)

    async with ClaudeSDKClient(options) as client:
        # Disable noisy built-in MCP servers
        try:
            await client.toggle_mcp_server(
                "claude.ai Google Calendar", enabled=False
            )
            await client.toggle_mcp_server("claude.ai Gmail", enabled=False)
        except Exception:
            pass  # Not critical if these fail

        print("  Lux ready.", flush=True)
        print("  Waiting for commands...\n", flush=True)

        server = await asyncio.start_unix_server(
            lambda r, w: _handle_client(r, w, client, options),
            path=str(SOCKET_PATH),
        )

        # Handle shutdown gracefully
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()

        def _shutdown():
            print("\nShutting down...", flush=True)
            stop.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _shutdown)

        # Start background tasks
        alert_task = asyncio.create_task(alert_loop())
        scheduler_task = asyncio.create_task(scheduler_loop())

        async with server:
            await stop.wait()

        # Shutdown background tasks
        await _stop_breathing()
        for task in (alert_task, scheduler_task):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Cleanup
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        if PID_FILE.exists():
            PID_FILE.unlink()


def main() -> None:
    try:
        asyncio.run(run_daemon())
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
