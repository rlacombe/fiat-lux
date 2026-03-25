"""Hey Lux daemon — persistent ClaudeSDKClient with Unix socket interface.

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

from heylux.alerts import alert_loop
from heylux.routines import ALL_ROUTINE_TOOLS
from heylux.scheduler import scheduler_loop
from heylux.shortcuts import (
    try_shortcut,
    SHORTCUT_BREATHE_START,
    SHORTCUT_BREATHE_STOP,
    SHORTCUT_CANDLE_START,
)
from heylux.mcp.ambient import (
    ALL_AMBIENT_TOOLS,
    start_candle,
    start_breathe,
    stop_ambient,
)
from heylux.mcp.calendar_tools import ALL_CALENDAR_TOOLS
from heylux.mcp.circadian import get_circadian_recommendation, configure_light_map
from heylux.mcp.hue import ALL_HUE_TOOLS, get_lights_context
from heylux.mcp.memory import ALL_MEMORY_TOOLS, get_profile_context
from heylux.mcp.weather_tools import ALL_WEATHER_TOOLS
from heylux.weather import get_weather_context
from heylux.mcp.scheduler_tools import ALL_SCHEDULER_TOOLS

SOCKET_PATH = Path.home() / ".config" / "heylux" / "lux.sock"
PID_FILE = Path.home() / ".config" / "heylux" / "lux.pid"
BASE_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text()


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


VOICE_MODE_PROMPT = """

## Voice Mode Active
The user is speaking voice commands. Your response will be read aloud by TTS.
SPEED IS CRITICAL. Every word costs playback time.

RULES:
- Call tools FIRST. NO text before tools.
- After tools, reply with MAX 8 WORDS. Format: "<color/mood>. <send-off>!"
- Use parallel tool_use blocks when setting multiple lights (faster than sequential).
- NO emoji (spoken as words), NO markdown, NO light names.
- Vary send-offs.

GOOD: "Green and purple hues. Happy coding!"
GOOD: "Warm amber glow. Sweet dreams!"
GOOD: "Lights off. Goodnight!"
BAD: "Setting up..." (text before tools)
BAD: "Your room is now glowing in warm golden ambers and soft peach..." (too long)
BAD: "💜💚" (emoji spoken as words)
"""

# Use Haiku in voice mode for speed
VOICE_MODEL = "claude-haiku-4-5-20251001"


_original_model: str | None = None


def _inject_voice_mode(options: ClaudeAgentOptions) -> None:
    """Switch to Haiku and add voice instructions."""
    global _original_model
    if VOICE_MODE_PROMPT not in options.system_prompt:
        options.system_prompt += VOICE_MODE_PROMPT
    _original_model = options.model
    options.model = VOICE_MODEL
    options.max_turns = 3  # keep it fast


def _remove_voice_mode(options: ClaudeAgentOptions) -> None:
    """Restore normal model and remove voice instructions."""
    global _original_model
    options.system_prompt = options.system_prompt.replace(VOICE_MODE_PROMPT, "")
    if _original_model is not None:
        options.model = _original_model
        _original_model = None
    options.max_turns = 10  # restore normal


def _build_options() -> ClaudeAgentOptions:
    all_sdk_tools = [
        get_circadian_recommendation,
        configure_light_map,
        *ALL_HUE_TOOLS,
        *ALL_AMBIENT_TOOLS,
        *ALL_MEMORY_TOOLS,
        *ALL_ROUTINE_TOOLS,
        *ALL_CALENDAR_TOOLS,
        *ALL_SCHEDULER_TOOLS,
        *ALL_WEATHER_TOOLS,
    ]
    heylux_tools = create_sdk_mcp_server(
        name="heylux",
        version="0.1.0",
        tools=all_sdk_tools,
    )
    # Pre-load tool names so Claude doesn't need ToolSearch
    tool_names = [f"mcp__heylux__{t.name}" for t in all_sdk_tools]
    return ClaudeAgentOptions(
        system_prompt=_build_system_prompt(),
        mcp_servers={"heylux": heylux_tools},
        tools=tool_names,
        allowed_tools=["mcp__heylux__*"],
        permission_mode="acceptEdits",
        max_turns=10,
        setting_sources=[],
    )


def _resolve_light_ids(light_name: str) -> list[int] | None:
    """Resolve a light name to IDs. Returns None for all lights."""
    if not light_name:
        return None
    try:
        from heylux.mcp.hue import _get_bridge, _normalize
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
    """Handle ambient mode start/stop signals from shortcuts.

    Uses shared state from heylux.mcp.ambient so both shortcuts and
    Claude tool calls manage the same ambient task.
    """
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
        await start_breathe(light_ids)
        return f"Breathing mode started{light_label}. Say 'stop' to end."

    if shortcut_result == SHORTCUT_CANDLE_START:
        await start_candle(light_ids, fade_out_minutes=fade_minutes)
        return f"Candle mode started{light_label}{fade_label}. Say 'stop' to end."

    if shortcut_result == SHORTCUT_BREATHE_STOP:
        was_running = await stop_ambient()
        if was_running:
            return "Ambient mode stopped. Lights restored."
        # Not in ambient mode — just turn everything off
        from heylux.shortcuts import _all_off
        return _all_off()

    # Any other shortcut: stop ambient mode if active
    from heylux.mcp.ambient import _ambient_task
    if _ambient_task is not None and not _ambient_task.done():
        await stop_ambient()

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
        voice_mode = request.get("voice", False)

        if not prompt:
            writer.write(json.dumps({"type": "done"}).encode() + b"\n")
            await writer.drain()
            return

        # Tier 1: shortcuts — instant, no LLM
        shortcut_result = try_shortcut(prompt)
        if shortcut_result is not None:
            # Handle breathing mode start/stop
            shortcut_result = await _handle_ambient(shortcut_result)

            # Check if the routine wants to start an ambient mode (candle/breathe)
            from heylux.routines import pop_pending_ambient
            pending = pop_pending_ambient()
            if pending:
                mode = pending["mode"]
                light_ids = pending["light_ids"]
                fade = pending.get("fade_out_minutes", 0)
                if mode == "candle":
                    await start_candle(light_ids, fade_out_minutes=fade)
                elif mode == "breathe":
                    await start_breathe(light_ids)

            writer.write(
                json.dumps({"type": "text", "text": shortcut_result}).encode() + b"\n"
            )
            writer.write(json.dumps({"type": "done"}).encode() + b"\n")
            await writer.drain()
            return

        # Any non-shortcut command stops breathing mode
        await stop_ambient()

        # Inject voice mode instructions if needed
        if voice_mode:
            _inject_voice_mode(options)
            # Prepend voice constraint directly to the prompt so Claude can't miss it
            prompt = (
                "[VOICE: Call tools FIRST with NO text before them. "
                "After tools, reply with MAX 8 words total. "
                "Format: '<color/mood>. <2-word send-off>!' "
                "Example: 'Warm amber glow. Sweet dreams!' "
                "If it matches a routine, use list_routines then apply it.]\n\n"
                + prompt
            )
        else:
            _remove_voice_mode(options)

        # Tier 2: Claude via persistent ClaudeSDKClient
        #
        # In voice mode, split text blocks into sentences so the GUI can
        # start TTS on the first sentence before Claude finishes generating.
        import re
        _sentence_re = re.compile(r'(?<=[.!?])\s+')

        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text") and block.text:
                        if voice_mode:
                            # Split into sentences for faster time-to-first-audio
                            sentences = _sentence_re.split(block.text)
                            for sentence in sentences:
                                sentence = sentence.strip()
                                if sentence:
                                    writer.write(
                                        json.dumps({"type": "text", "text": sentence}).encode()
                                        + b"\n"
                                    )
                                    await writer.drain()
                        else:
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
    """Start the Hey Lux daemon."""
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
        await stop_ambient()
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
