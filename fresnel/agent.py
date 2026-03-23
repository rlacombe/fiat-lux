"""Fresnel — A chronobiology-powered lighting agent for Philips Hue.

Usage:
    fresnel                          Interactive mode
    fresnel "make it cozy"           One-shot command
    fresnel circadian                Apply current circadian recommendation
    fresnel setup                    Guide Hue Bridge setup

Three execution tiers:
    1. Shortcuts  — regex-matched common commands, no LLM, <0.5s
    2. LLM engine — OpenAI-compatible API (LM Studio / OpenRouter), ~2-5s
    3. Agent SDK  — Claude Code via claude_agent_sdk (fallback), ~14s
"""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from fresnel.shortcuts import try_shortcut

load_dotenv()

console = Console()


def _run_prompt(prompt: str) -> None:
    """Route a prompt through the three execution tiers."""
    # Tier 1: direct shortcuts (no LLM)
    result = try_shortcut(prompt)
    if result is not None:
        console.print(result)
        return

    # Tier 2: lightweight LLM engine (OpenAI-compatible)
    engine = os.getenv("FRESNEL_ENGINE", "llm")

    if engine == "llm":
        from fresnel.engine import chat

        try:
            response = chat(prompt)
            console.print(Markdown(response))
        except Exception as e:
            console.print(f"[yellow]LLM engine error: {e}[/yellow]")
            console.print("[dim]Falling back to Agent SDK...[/dim]")
            _run_agent_sdk(prompt)
        return

    # Tier 3: full Agent SDK
    _run_agent_sdk(prompt)


def _run_agent_sdk(prompt: str) -> None:
    """Run via Claude Agent SDK (heaviest, most capable)."""
    import asyncio

    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        SystemMessage,
        create_sdk_mcp_server,
        query,
    )

    from fresnel.tools.circadian import get_circadian_recommendation
    from fresnel.tools.hue import ALL_HUE_TOOLS, get_lights_context
    from fresnel.tools.memory import ALL_MEMORY_TOOLS, get_profile_context
    from pathlib import Path

    BASE_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text()

    parts = [BASE_SYSTEM_PROMPT]
    profile_ctx = get_profile_context()
    if profile_ctx:
        parts.append(profile_ctx)
    lights_ctx = get_lights_context()
    if lights_ctx:
        parts.append(lights_ctx)
    system_prompt = "\n\n".join(parts)

    fresnel_tools = create_sdk_mcp_server(
        name="fresnel",
        version="0.1.0",
        tools=[get_circadian_recommendation, *ALL_HUE_TOOLS, *ALL_MEMORY_TOOLS],
    )

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={"fresnel": fresnel_tools},
        allowed_tools=["mcp__fresnel__*"],
        permission_mode="acceptEdits",
        max_turns=10,
        setting_sources=[],
    )

    async def _stream():
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text") and block.text:
                        console.print(Markdown(block.text))
                    elif hasattr(block, "name"):
                        console.print(f"[dim]→ {block.name}[/dim]")
            elif isinstance(message, ResultMessage):
                if message.subtype != "success":
                    console.print(f"[red]Error: {message.subtype}[/red]")

    asyncio.run(_stream())


def _interactive() -> None:
    """Run Fresnel in interactive mode."""
    console.print(
        "[bold]Fresnel[/bold] — your chronobiology-powered lighting assistant\n"
        "[dim]Type a command, or 'quit' to exit.[/dim]\n"
    )

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        console.print()
        _run_prompt(user_input)
        console.print()


def main() -> None:
    """CLI entry point."""
    args = sys.argv[1:]

    if not args:
        _interactive()
    elif args[0] == "setup":
        # Setup always uses Agent SDK for the full conversational flow
        os.environ["FRESNEL_ENGINE"] = "agent-sdk"
        _run_prompt(
            "Help me set up my Philips Hue Bridge. Walk me through discovering "
            "the bridge on my network, pressing the link button, and verifying "
            "the connection. List all my lights when done."
        )
    else:
        prompt = " ".join(args)
        _run_prompt(prompt)


if __name__ == "__main__":
    main()
