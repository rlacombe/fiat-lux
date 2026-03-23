"""MCP tools for calendar alert setup — lets Lux drive the flow via conversation."""

from typing import Any

from claude_agent_sdk import tool

from fiat_lux.calendar import (
    icalbuddy_available,
    install_icalbuddy,
    list_calendars,
    _load_config,
    _save_config,
)


def _text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _error(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


@tool(
    "setup_calendar_alerts",
    "Check if calendar alerts are ready. Installs icalBuddy (via Homebrew) if "
    "missing, then lists all available calendars from macOS Calendar.app. "
    "Returns the list so you can ask the user which ones to monitor.",
    {},
)
async def setup_calendar_alerts(args: dict[str, Any]) -> dict[str, Any]:
    # Step 1: ensure icalBuddy
    if not icalbuddy_available():
        if not install_icalbuddy():
            return _error(
                "icalBuddy is not installed and automatic install failed. "
                "The user can install it manually with: brew install ical-buddy"
            )

    # Step 2: list calendars
    calendars = list_calendars()
    if not calendars:
        return _error("No calendars found in macOS Calendar.app.")

    # Step 3: check current config
    config = _load_config()
    current = config.get("calendars", [])

    lines = ["**Available calendars:**\n"]
    for i, cal in enumerate(calendars, 1):
        name = cal["name"]
        active = " (active)" if name in current else ""
        lines.append(f"{i}. {name} ({cal.get('type', '?')}){active}")

    if current:
        lines.append(f"\n**Currently monitoring:** {', '.join(current)}")
    else:
        lines.append("\n**No calendars configured yet.**")

    lines.append(
        "\nAsk the user which calendars to monitor, then call "
        "save_calendar_config with their choices."
    )
    return _text("\n".join(lines))


@tool(
    "save_calendar_config",
    "Save which calendars to monitor for meeting alerts. Pass the exact "
    "calendar names the user selected. This enables the background alert "
    "loop (amber pulse 5 min before, blue pulse 15 sec before meetings).",
    {
        "type": "object",
        "properties": {
            "calendars": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of calendar names to monitor (exact names from setup_calendar_alerts).",
            },
        },
        "required": ["calendars"],
    },
)
async def save_calendar_config(args: dict[str, Any]) -> dict[str, Any]:
    names = args["calendars"]
    if not names:
        return _error("No calendars specified.")

    # Validate against available calendars
    available = {cal["name"] for cal in list_calendars()}
    unknown = [n for n in names if n not in available]
    if unknown:
        return _error(
            f"Unknown calendar(s): {', '.join(unknown)}. "
            "Use setup_calendar_alerts to see available names."
        )

    config = _load_config()
    config["calendars"] = names
    _save_config(config)

    return _text(
        f"Saved {len(names)} calendar(s) for meeting alerts:\n"
        + "\n".join(f"- {n}" for n in names)
        + "\n\nThe daemon will pulse your desk lamp amber 5 min before "
        "and blue 15 sec before each meeting. "
        "Alerts will activate on next daemon restart."
    )


@tool(
    "set_alert_lights",
    "Configure which lights pulse for calendar meeting alerts. "
    "Pass light names to use for alerts, or ['all'] for all lights (room-wide). "
    "Defaults to all lights if not configured.",
    {
        "type": "object",
        "properties": {
            "lights": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Light names to use for meeting alerts. Use ['all'] for all lights.",
            },
        },
        "required": ["lights"],
    },
)
async def set_alert_lights(args: dict[str, Any]) -> dict[str, Any]:
    lights = args["lights"]
    config = _load_config()
    config["alert_lights"] = lights
    _save_config(config)
    if lights == ["all"]:
        return _text("Meeting alerts will pulse all lights (room-wide).")
    return _text(
        f"Meeting alerts will pulse: {', '.join(lights)}.\n"
        "All other lights stay untouched during alerts."
    )


ALL_CALENDAR_TOOLS = [setup_calendar_alerts, save_calendar_config, set_alert_lights]
