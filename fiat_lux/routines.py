"""Fiat-Lux routines — named lighting presets, configurable through conversation.

Stored in ~/.config/fiat_lux/routines.json. Each routine defines which lights
to turn on/off and their settings. Triggered instantly via shortcuts.
"""

import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool

from fiat_lux.mcp.hue import _get_bridge, _normalize

CONFIG_DIR = Path.home() / ".config" / "fiat_lux"
ROUTINES_FILE = CONFIG_DIR / "routines.json"

# Default routines — seeded on first use, then user-customizable
DEFAULT_ROUTINES: dict[str, Any] = {
    "bedtime": {
        "description": "Reading in bed — only nightstand, warm and comfortable",
        "lights_on": {
            "Night stand": {"brightness_pct": 60, "kelvin": 2200},
        },
        "lights_off": ["Ceiling lamp 1", "Ceiling lamp 2", "Ceiling lamp 3", "Desk lamp", "Lantern"],
        "transition_seconds": 3,
    },
    "goodnight": {
        "description": "Everything off — sleep time",
        "lights_on": {},
        "lights_off": ["all"],
        "transition_seconds": 2,
    },
    "morning": {
        "description": "Wake up — ceiling and desk, cool and bright",
        "lights_on": {
            "Ceiling lamp 1": {"brightness_pct": 80, "kelvin": 5000},
            "Ceiling lamp 2": {"brightness_pct": 80, "kelvin": 5000},
            "Ceiling lamp 3": {"brightness_pct": 80, "kelvin": 5000},
            "Desk lamp": {"brightness_pct": 70, "kelvin": 4500},
        },
        "lights_off": ["Night stand", "Lantern"],
        "transition_seconds": 5,
    },
    "focus": {
        "description": "Deep work — ceiling and desk at peak alertness",
        "lights_on": {
            "Ceiling lamp 1": {"brightness_pct": 95, "kelvin": 5500},
            "Ceiling lamp 2": {"brightness_pct": 95, "kelvin": 5500},
            "Ceiling lamp 3": {"brightness_pct": 95, "kelvin": 5500},
            "Desk lamp": {"brightness_pct": 90, "kelvin": 5500},
        },
        "lights_off": ["Night stand", "Lantern"],
        "transition_seconds": 2,
    },
    "reading": {
        "description": "Comfortable reading — nightstand and desk, warm white",
        "lights_on": {
            "Night stand": {"brightness_pct": 70, "kelvin": 3000},
            "Desk lamp": {"brightness_pct": 60, "kelvin": 3000},
        },
        "lights_off": ["Ceiling lamp 1", "Ceiling lamp 2", "Ceiling lamp 3", "Lantern"],
        "transition_seconds": 2,
    },
    "relax": {
        "description": "Evening wind-down — low warm light from lantern and nightstand",
        "lights_on": {
            "Lantern": {"brightness_pct": 40, "kelvin": 2200},
            "Night stand": {"brightness_pct": 30, "kelvin": 2200},
        },
        "lights_off": ["Ceiling lamp 1", "Ceiling lamp 2", "Ceiling lamp 3", "Desk lamp"],
        "transition_seconds": 3,
    },
}


def _load_routines() -> dict[str, Any]:
    if ROUTINES_FILE.exists():
        return json.loads(ROUTINES_FILE.read_text())
    # Seed defaults on first use
    _save_routines(DEFAULT_ROUTINES)
    return DEFAULT_ROUTINES


def _save_routines(routines: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ROUTINES_FILE.write_text(json.dumps(routines, indent=2))


def list_routines() -> dict[str, str]:
    """Return routine names and descriptions."""
    routines = _load_routines()
    return {name: r.get("description", "") for name, r in routines.items()}


def run_routine(name: str) -> str | None:
    """Execute a routine by name. Returns status text, or None if not found."""
    routines = _load_routines()
    routine = routines.get(name.lower())
    if routine is None:
        return None

    try:
        b = _get_bridge()
    except RuntimeError as e:
        return f"Error: {e}"

    transition = routine.get("transition_seconds", 0.4)
    transitiontime = round(transition * 10)

    # Build name→id map with normalized names
    name_map = {}
    for light in b.lights:
        name_map[_normalize(light.name).lower()] = light.light_id

    all_ids = set(name_map.values())

    # Turn off lights
    off_lights = routine.get("lights_off", [])
    off_ids = set()
    if off_lights == ["all"] or "all" in off_lights:
        off_ids = all_ids
    else:
        for lname in off_lights:
            lid = name_map.get(_normalize(lname).lower())
            if lid:
                off_ids.add(lid)

    # Turn on lights with settings
    on_lights = routine.get("lights_on", {})
    on_ids = set()
    for lname, settings in on_lights.items():
        lid = name_map.get(_normalize(lname).lower())
        if lid is None:
            continue
        on_ids.add(lid)

        cmd: dict[str, Any] = {"on": True, "transitiontime": transitiontime}
        if "brightness_pct" in settings:
            cmd["bri"] = round(settings["brightness_pct"] * 254 / 100)
        if "kelvin" in settings:
            cmd["ct"] = round(1_000_000 / settings["kelvin"])
        if "hue" in settings:
            cmd["hue"] = int(settings["hue"])
        if "saturation" in settings:
            cmd["sat"] = int(settings["saturation"])

        b.set_light(lid, cmd)

    # Turn off (but don't turn off lights that were explicitly turned on)
    for lid in off_ids - on_ids:
        b.set_light(lid, {"on": False, "transitiontime": transitiontime})

    desc = routine.get("description", name)
    return f"{name.capitalize()}: {desc}"


# ---------------------------------------------------------------------------
# MCP tools for the LLM to manage routines
# ---------------------------------------------------------------------------


def _text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


@tool(
    "list_routines",
    "List all saved lighting routines with their descriptions.",
    {},
)
async def list_routines_tool(args: dict[str, Any]) -> dict[str, Any]:
    routines = list_routines()
    if not routines:
        return _text("No routines saved yet.")
    lines = ["**Saved Routines:**"]
    for name, desc in routines.items():
        lines.append(f"  - **{name}**: {desc}")
    return _text("\n".join(lines))


@tool(
    "save_routine",
    "Create or update a lighting routine. Routines define which lights to turn "
    "on (with brightness/color settings) and which to turn off. Users can trigger "
    "them instantly by name.",
    {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Routine name (e.g. 'bedtime', 'focus', 'movie').",
            },
            "description": {
                "type": "string",
                "description": "Short description of what this routine does.",
            },
            "lights_on": {
                "type": "object",
                "description": (
                    "Lights to turn on. Keys are light names, values are objects "
                    "with optional brightness_pct (0-100), kelvin (2000-6500), "
                    "hue (0-65535), saturation (0-254)."
                ),
            },
            "lights_off": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Light names to turn off. Use 'all' for all lights not in lights_on.",
            },
            "transition_seconds": {
                "type": "number",
                "description": "Fade duration. Default 2.",
            },
        },
        "required": ["name", "description", "lights_on", "lights_off"],
    },
)
async def save_routine_tool(args: dict[str, Any]) -> dict[str, Any]:
    routines = _load_routines()
    name = args["name"].lower()

    # Validate light names against the bridge
    warnings = []
    try:
        b = _get_bridge()
        known = {_normalize(l.name).lower() for l in b.lights}
        for lname in list(args["lights_on"].keys()) + args["lights_off"]:
            if lname.lower() != "all" and _normalize(lname).lower() not in known:
                warnings.append(lname)
    except RuntimeError:
        pass  # Bridge not configured — skip validation

    routines[name] = {
        "description": args["description"],
        "lights_on": args["lights_on"],
        "lights_off": args["lights_off"],
        "transition_seconds": args.get("transition_seconds", 2),
    }
    _save_routines(routines)

    msg = f"Saved routine '{name}'. Trigger it anytime by typing '{name}'."
    if warnings:
        msg += f"\n\nWarning: these light names weren't found on the bridge: {', '.join(warnings)}"
    return _text(msg)


@tool(
    "delete_routine",
    "Delete a saved lighting routine.",
    {"name": str},
)
async def delete_routine_tool(args: dict[str, Any]) -> dict[str, Any]:
    routines = _load_routines()
    name = args["name"].lower()
    if name in routines:
        del routines[name]
        _save_routines(routines)
        return _text(f"Deleted routine '{name}'.")
    return _text(f"Routine '{name}' not found.")


ALL_ROUTINE_TOOLS = [
    list_routines_tool,
    save_routine_tool,
    delete_routine_tool,
]
