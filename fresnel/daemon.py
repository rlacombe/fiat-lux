"""Fresnel daemon — persistent ClaudeSDKClient with Unix socket interface.

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

from fresnel.routines import ALL_ROUTINE_TOOLS
from fresnel.shortcuts import try_shortcut
from fresnel.tools.circadian import get_circadian_recommendation
from fresnel.tools.hue import ALL_HUE_TOOLS, get_lights_context
from fresnel.tools.memory import ALL_MEMORY_TOOLS, get_profile_context

SOCKET_PATH = Path.home() / ".config" / "fresnel" / "fresnel.sock"
PID_FILE = Path.home() / ".config" / "fresnel" / "fresnel.pid"
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
    return "\n\n".join(parts)


def _build_options() -> ClaudeAgentOptions:
    all_sdk_tools = [
        get_circadian_recommendation,
        *ALL_HUE_TOOLS,
        *ALL_MEMORY_TOOLS,
        *ALL_ROUTINE_TOOLS,
    ]
    fresnel_tools = create_sdk_mcp_server(
        name="fresnel",
        version="0.1.0",
        tools=all_sdk_tools,
    )
    # Pre-load tool names so Claude doesn't need ToolSearch
    tool_names = [f"mcp__fresnel__{t.name}" for t in all_sdk_tools]
    return ClaudeAgentOptions(
        system_prompt=_build_system_prompt(),
        mcp_servers={"fresnel": fresnel_tools},
        tools=tool_names,
        allowed_tools=["mcp__fresnel__*"],
        permission_mode="acceptEdits",
        max_turns=10,
        setting_sources=[],
    )


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    client: ClaudeSDKClient,
):
    """Handle a single CLI connection."""
    try:
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
            writer.write(
                json.dumps({"type": "text", "text": shortcut_result}).encode() + b"\n"
            )
            writer.write(json.dumps({"type": "done"}).encode() + b"\n")
            await writer.drain()
            return

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
    """Start the Fresnel daemon."""
    # Clean up stale socket
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

    # Write PID file
    PID_FILE.write_text(str(os.getpid()))

    options = _build_options()

    print("Lux daemon starting...")
    print(f"  Socket: {SOCKET_PATH}")

    async with ClaudeSDKClient(options) as client:
        # Disable noisy built-in MCP servers
        try:
            await client.toggle_mcp_server(
                "claude.ai Google Calendar", enabled=False
            )
            await client.toggle_mcp_server("claude.ai Gmail", enabled=False)
        except Exception:
            pass  # Not critical if these fail

        print("  Lux ready.")
        print("  Waiting for commands...\n")

        server = await asyncio.start_unix_server(
            lambda r, w: _handle_client(r, w, client),
            path=str(SOCKET_PATH),
        )

        # Handle shutdown gracefully
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()

        def _shutdown():
            print("\nShutting down...")
            stop.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _shutdown)

        async with server:
            await stop.wait()

        # Cleanup
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        if PID_FILE.exists():
            PID_FILE.unlink()


def main() -> None:
    asyncio.run(run_daemon())


if __name__ == "__main__":
    main()
