"""Fiat-Lux user memory — persistent user profile across sessions.

Stores user preferences, room layout, chronotype, and anything else
Lux learns through conversation. Lives in ~/.config/fiat_lux/user.json.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool

CONFIG_DIR = Path.home() / ".config" / "fiat_lux"
USER_FILE = CONFIG_DIR / "user.json"


def _load_profile() -> dict[str, Any]:
    if USER_FILE.exists():
        return json.loads(USER_FILE.read_text())
    return {}


def _save_profile(profile: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    USER_FILE.write_text(json.dumps(profile, indent=2))


def _text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def get_profile_context() -> str:
    """Return the user profile as text to inject into system prompt.

    Called at startup, not as a tool.
    """
    profile = _load_profile()
    if not profile:
        return ""

    lines = ["## Known User Profile"]
    for key, entry in profile.items():
        if isinstance(entry, dict) and "value" in entry:
            lines.append(f"- **{key}**: {entry['value']}")
        else:
            lines.append(f"- **{key}**: {entry}")
    return "\n".join(lines)


@tool(
    "get_user_profile",
    "Retrieve everything Lux knows about the user from past conversations. "
    "Returns the full user profile. Call this when you need to personalize advice "
    "or check what you already know before asking the user again.",
    {},
)
async def get_user_profile(args: dict[str, Any]) -> dict[str, Any]:
    profile = _load_profile()
    if not profile:
        return _text(
            "No user profile saved yet. This appears to be a new user. "
            "Ask them about their name, room layout, sleep habits, and "
            "typical schedule so you can give personalized recommendations."
        )

    lines = []
    for key, entry in profile.items():
        if isinstance(entry, dict) and "value" in entry:
            lines.append(f"- **{key}**: {entry['value']}")
        else:
            lines.append(f"- **{key}**: {entry}")
    return _text("User profile:\n" + "\n".join(lines))


@tool(
    "save_user_info",
    "Save something you learned about the user for future sessions. Use this "
    "whenever the user shares personal details relevant to their lighting needs: "
    "name, room layout, sleep sensitivity, chronotype, work schedule, preferences. "
    "Each key is a topic (e.g. 'name', 'sleep_sensitivity', 'room_layout'). "
    "Overwrites previous values for the same key.",
    {"key": str, "value": str},
)
async def save_user_info(args: dict[str, Any]) -> dict[str, Any]:
    key = args["key"]
    value = args["value"]

    profile = _load_profile()
    profile[key] = {
        "value": value,
        "updated": datetime.now().isoformat(),
    }
    _save_profile(profile)
    return _text(f"Saved: {key} = {value}")


@tool(
    "forget_user_info",
    "Remove a specific piece of information from the user's profile. "
    "Use when the user asks you to forget something or correct outdated info.",
    {"key": str},
)
async def forget_user_info(args: dict[str, Any]) -> dict[str, Any]:
    key = args["key"]
    profile = _load_profile()
    if key in profile:
        del profile[key]
        _save_profile(profile)
        return _text(f"Removed '{key}' from profile.")
    return _text(f"No entry '{key}' found in profile.")


ALL_MEMORY_TOOLS = [
    get_user_profile,
    save_user_info,
    forget_user_info,
]
