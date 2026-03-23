"""Philips Hue light control tools.

Wraps the phue library to provide MCP tools for Fresnel's agent.
Handles bridge discovery, pairing, and all light/group/scene control.
"""

import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool
from phue import Bridge, PhueRegistrationException

# Fresnel stores its config separately from phue's default ~/.python_hue
CONFIG_DIR = Path.home() / ".config" / "fresnel"
CONFIG_FILE = CONFIG_DIR / "hue.json"


def _load_config() -> dict[str, Any]:
    """Load saved Hue bridge config."""
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def _save_config(config: dict[str, Any]) -> None:
    """Persist Hue bridge config."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def _get_bridge() -> Bridge:
    """Get a connected Bridge instance from saved config.

    Raises RuntimeError if no bridge is configured.
    """
    config = _load_config()
    ip = config.get("bridge_ip")
    username = config.get("username")
    if not ip or not username:
        raise RuntimeError(
            "No Hue Bridge configured. Use the pair_hue_bridge tool first."
        )
    return Bridge(ip, username)


def _normalize(name: str) -> str:
    """Normalize smart quotes and whitespace for matching."""
    return name.replace("\u2018", "'").replace("\u2019", "'").strip()


def _find_group_id(b: Bridge, group_name: str) -> int | None:
    """Find a group by name with fuzzy quote matching."""
    target = _normalize(group_name).lower()
    for gid, group in b.get_group().items():
        if _normalize(group["name"]).lower() == target:
            return int(gid)
    return None


def _text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _error(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


# ---------------------------------------------------------------------------
# Bridge setup
# ---------------------------------------------------------------------------


@tool(
    "pair_hue_bridge",
    "Pair with a Philips Hue Bridge. The user must press the physical link button "
    "on the bridge BEFORE calling this tool. Requires the bridge IP address. "
    "On success, saves credentials so future calls work automatically.",
    {"bridge_ip": str},
)
async def pair_hue_bridge(args: dict[str, Any]) -> dict[str, Any]:
    ip = args["bridge_ip"]
    try:
        # phue attempts registration on Bridge() init
        b = Bridge(ip, config_file_path=str(CONFIG_DIR / ".python_hue"))
        # If we get here, pairing succeeded (or was already paired)
        username = b.username
        _save_config({"bridge_ip": ip, "username": username})

        # Verify by listing lights
        lights = b.get_light_objects("name")
        light_names = list(lights.keys())
        return _text(
            f"Successfully paired with bridge at {ip}!\n"
            f"Found {len(light_names)} light(s): {', '.join(light_names)}\n"
            f"Credentials saved to {CONFIG_FILE}"
        )
    except PhueRegistrationException:
        return _error(
            "The link button was not pressed. "
            "Ask the user to press the button on the Hue Bridge, "
            "then call this tool again within 30 seconds."
        )
    except Exception as e:
        return _error(f"Failed to connect to bridge at {ip}: {e}")


@tool(
    "get_hue_status",
    "Check the current Hue Bridge connection status and list all available lights, "
    "groups, and scenes. Use this to understand what's available before making changes.",
    {},
)
async def get_hue_status(args: dict[str, Any]) -> dict[str, Any]:
    try:
        b = _get_bridge()
    except RuntimeError as e:
        return _error(str(e))

    try:
        lines = ["**Hue Bridge Status**\n"]

        # Lights
        lines.append("**Lights:**")
        for light in b.lights:
            state = "on" if light.on else "off"
            bri = light.brightness
            ct = getattr(light, "colortemp", None)
            ct_str = f", {ct} mireds (~{round(1_000_000 / ct)}K)" if ct else ""
            lines.append(
                f"  - {light.name} (id={light.light_id}): "
                f"{state}, brightness={bri}/254{ct_str}"
            )

        # Groups
        groups = b.get_group()
        if groups:
            lines.append("\n**Groups/Rooms:**")
            for gid, group in groups.items():
                lines.append(
                    f"  - {group['name']} (id={gid}): "
                    f"lights={group.get('lights', [])}"
                )

        # Scenes
        scenes = b.get_scene()
        if scenes:
            lines.append("\n**Scenes:**")
            seen = set()
            for sid, scene in scenes.items():
                name = scene.get("name", sid)
                if name not in seen:
                    seen.add(name)
                    lines.append(f"  - {name} (id={sid})")

        return _text("\n".join(lines))
    except Exception as e:
        return _error(f"Failed to get bridge status: {e}")


# ---------------------------------------------------------------------------
# Light control
# ---------------------------------------------------------------------------


@tool(
    "set_lights",
    "Set the state of one or more Hue lights. Can control brightness, color "
    "temperature, color, and on/off state. Accepts light names or IDs. "
    "Use transition_time for smooth fades (in seconds).",
    {
        "type": "object",
        "properties": {
            "lights": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Light names or IDs to control. Use 'all' for all lights.",
            },
            "on": {
                "type": "boolean",
                "description": "Turn lights on (true) or off (false).",
            },
            "brightness_pct": {
                "type": "number",
                "description": "Brightness as percentage (0-100). Converted to 0-254 for Hue.",
            },
            "kelvin": {
                "type": "number",
                "description": "Color temperature in Kelvin (2000-6500).",
            },
            "hue": {
                "type": "number",
                "description": "Hue value (0-65535). Red=0, Green=~21845, Blue=~43690.",
            },
            "saturation": {
                "type": "number",
                "description": "Saturation (0-254). 0=white, 254=full color.",
            },
            "transition_seconds": {
                "type": "number",
                "description": "Fade duration in seconds. Default 0.4.",
            },
        },
        "required": ["lights"],
    },
)
async def set_lights(args: dict[str, Any]) -> dict[str, Any]:
    try:
        b = _get_bridge()
    except RuntimeError as e:
        return _error(str(e))

    light_names = args["lights"]

    # Resolve "all"
    if light_names == ["all"]:
        targets = [l.light_id for l in b.lights]
    else:
        targets = []
        name_map = {l.name.lower(): l.light_id for l in b.lights}
        for name in light_names:
            if name.isdigit():
                targets.append(int(name))
            elif name.lower() in name_map:
                targets.append(name_map[name.lower()])
            else:
                return _error(
                    f"Unknown light '{name}'. "
                    f"Available: {', '.join(name_map.keys())}"
                )

    # Build command dict
    cmd: dict[str, Any] = {}

    if "on" in args:
        cmd["on"] = args["on"]

    if "brightness_pct" in args:
        pct = max(0, min(100, args["brightness_pct"]))
        cmd["bri"] = round(pct * 254 / 100)
        if pct > 0 and "on" not in cmd:
            cmd["on"] = True

    if "kelvin" in args:
        kelvin = max(2000, min(6500, args["kelvin"]))
        cmd["ct"] = round(1_000_000 / kelvin)

    if "hue" in args:
        cmd["hue"] = int(args["hue"])

    if "saturation" in args:
        cmd["sat"] = int(args["saturation"])

    transition = args.get("transition_seconds", 0.4)
    cmd["transitiontime"] = round(transition * 10)

    if not cmd or cmd == {"transitiontime": round(0.4 * 10)}:
        return _error("No changes specified. Set at least one of: on, brightness_pct, kelvin, hue, saturation.")

    try:
        for light_id in targets:
            b.set_light(light_id, cmd)

        changes = []
        if "on" in cmd:
            changes.append("on" if cmd["on"] else "off")
        if "bri" in cmd:
            changes.append(f"brightness={args['brightness_pct']}%")
        if "ct" in cmd:
            changes.append(f"color temp={args['kelvin']}K")
        if "hue" in cmd:
            changes.append(f"hue={cmd['hue']}")
        if "sat" in cmd:
            changes.append(f"saturation={cmd['sat']}")

        return _text(
            f"Set {len(targets)} light(s): {', '.join(changes)} "
            f"(fade: {transition}s)"
        )
    except Exception as e:
        return _error(f"Failed to set lights: {e}")


@tool(
    "activate_scene",
    "Activate a Hue scene by name in a specific room/group.",
    {"group_name": str, "scene_name": str},
)
async def activate_scene(args: dict[str, Any]) -> dict[str, Any]:
    try:
        b = _get_bridge()
    except RuntimeError as e:
        return _error(str(e))

    group_name = args["group_name"]
    scene_name = args["scene_name"]

    try:
        # Find the group ID ourselves (phue's run_scene has smart-quote issues)
        group_id = _find_group_id(b, group_name)
        if group_id is None:
            groups = [g["name"] for g in b.get_group().values()]
            return _error(
                f"Unknown group '{group_name}'. Available: {', '.join(groups)}"
            )

        # Find matching scene
        target_scene = _normalize(scene_name).lower()
        for sid, scene in b.get_scene().items():
            if _normalize(scene.get("name", "")).lower() == target_scene:
                b.activate_scene(group_id, sid)
                return _text(f"Activated scene '{scene_name}' in '{group_name}'.")

        return _error(
            f"Could not find scene '{scene_name}'. "
            "Use get_hue_status to see available scenes."
        )
    except Exception as e:
        return _error(f"Failed to activate scene: {e}")


@tool(
    "set_group",
    "Control all lights in a Hue group/room at once. Same options as set_lights.",
    {
        "type": "object",
        "properties": {
            "group_name": {
                "type": "string",
                "description": "Name of the group/room to control.",
            },
            "on": {"type": "boolean"},
            "brightness_pct": {"type": "number"},
            "kelvin": {"type": "number"},
            "transition_seconds": {"type": "number"},
        },
        "required": ["group_name"],
    },
)
async def set_group(args: dict[str, Any]) -> dict[str, Any]:
    try:
        b = _get_bridge()
    except RuntimeError as e:
        return _error(str(e))

    group_name = args["group_name"]
    group_id = _find_group_id(b, group_name)
    if group_id is None:
        groups = [g["name"] for g in b.get_group().values()]
        return _error(
            f"Unknown group '{group_name}'. Available: {', '.join(groups)}"
        )

    cmd: dict[str, Any] = {}

    if "on" in args:
        cmd["on"] = args["on"]

    if "brightness_pct" in args:
        pct = max(0, min(100, args["brightness_pct"]))
        cmd["bri"] = round(pct * 254 / 100)
        if pct > 0 and "on" not in cmd:
            cmd["on"] = True

    if "kelvin" in args:
        kelvin = max(2000, min(6500, args["kelvin"]))
        cmd["ct"] = round(1_000_000 / kelvin)

    transition = args.get("transition_seconds", 0.4)
    cmd["transitiontime"] = round(transition * 10)

    try:
        b.set_group(group_id, cmd)
        return _text(f"Updated group '{group_name}'.")
    except Exception as e:
        return _error(f"Failed to set group: {e}")


# All tools to register with the MCP server
ALL_HUE_TOOLS = [
    pair_hue_bridge,
    get_hue_status,
    set_lights,
    activate_scene,
    set_group,
]
