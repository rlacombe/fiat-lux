"""Lightweight LLM engine using OpenAI-compatible API.

Works with LM Studio (local), OpenRouter, OpenAI, or any provider
that speaks the OpenAI chat completions + tool use protocol.
No subprocess, no MCP, no Bun — just HTTP.
"""

import json
import os
from typing import Any

from openai import OpenAI

from fresnel.tools.hue import (
    _find_group_id,
    _get_bridge,
    _load_config,
    _normalize,
    _save_config,
    get_lights_context,
    CONFIG_DIR,
)
from fresnel.tools.memory import get_profile_context, _load_profile, _save_profile
from fresnel.tools.circadian import get_circadian_state

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_circadian_recommendation",
            "description": (
                "Get the optimal lighting recommendation for the current time "
                "based on circadian science. Returns color temperature, brightness, "
                "which lights should be active, and the current mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_override": {
                        "type": "string",
                        "description": "Optional time in HH:MM format. Defaults to now.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_lights",
            "description": (
                "Set the state of one or more Hue lights. Control brightness, "
                "color temperature, color, and on/off. Accepts light names or IDs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lights": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Light names or IDs. Use 'all' for all lights.",
                    },
                    "on": {"type": "boolean", "description": "Turn on (true) or off (false)."},
                    "brightness_pct": {
                        "type": "number",
                        "description": "Brightness 0-100%.",
                    },
                    "kelvin": {
                        "type": "number",
                        "description": "Color temperature in Kelvin (2000-6500).",
                    },
                    "hue": {
                        "type": "number",
                        "description": "Hue 0-65535. Red=0, Green=~21845, Blue=~43690.",
                    },
                    "saturation": {
                        "type": "number",
                        "description": "Saturation 0-254. 0=white, 254=full color.",
                    },
                    "transition_seconds": {
                        "type": "number",
                        "description": "Fade duration in seconds. Default 0.4.",
                    },
                },
                "required": ["lights"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_group",
            "description": "Control all lights in a Hue group/room at once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_name": {"type": "string", "description": "Room/group name."},
                    "on": {"type": "boolean"},
                    "brightness_pct": {"type": "number"},
                    "kelvin": {"type": "number"},
                    "transition_seconds": {"type": "number"},
                },
                "required": ["group_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_scene",
            "description": "Activate a Hue scene by name in a specific room/group.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_name": {"type": "string"},
                    "scene_name": {"type": "string"},
                },
                "required": ["group_name", "scene_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hue_status",
            "description": (
                "List all lights with current state, groups, and scenes. "
                "Only call if you need live state — light names are already in the system prompt."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_info",
            "description": (
                "Save something learned about the user for future sessions. "
                "Key is a topic (name, room_layout, sleep_sensitivity, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pair_hue_bridge",
            "description": (
                "Pair with a Hue Bridge. User must press link button first. "
                "Only needed once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bridge_ip": {"type": "string", "description": "Bridge IP address."},
                },
                "required": ["bridge_ip"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _exec_get_circadian(args: dict) -> str:
    from datetime import datetime

    time_str = args.get("time_override", "")
    now = None
    if time_str:
        try:
            now = datetime.strptime(time_str, "%H:%M").replace(
                year=datetime.now().year,
                month=datetime.now().month,
                day=datetime.now().day,
            )
        except ValueError:
            return f"Invalid time format '{time_str}'. Use HH:MM."

    state = get_circadian_state(now)
    return (
        f"Mode: {state['mode_name']} at {state['time']}\n"
        f"Color temp: {state['kelvin']}K, Brightness: {state['brightness_pct']}%\n"
        f"Active lights: {', '.join(state['active_lights'])}"
    )


def _exec_set_lights(args: dict) -> str:
    b = _get_bridge()
    light_names = args["lights"]

    if light_names == ["all"]:
        targets = [l.light_id for l in b.lights]
    else:
        targets = []
        name_map = {_normalize(l.name).lower(): l.light_id for l in b.lights}
        for name in light_names:
            if name.isdigit():
                targets.append(int(name))
            elif _normalize(name).lower() in name_map:
                targets.append(name_map[_normalize(name).lower()])
            else:
                return f"Unknown light '{name}'. Available: {', '.join(name_map.keys())}"

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

    for light_id in targets:
        b.set_light(light_id, cmd)

    return f"Set {len(targets)} light(s)."


def _exec_set_group(args: dict) -> str:
    b = _get_bridge()
    group_id = _find_group_id(b, args["group_name"])
    if group_id is None:
        groups = [g["name"] for g in b.get_group().values()]
        return f"Unknown group '{args['group_name']}'. Available: {', '.join(groups)}"

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

    b.set_group(group_id, cmd)
    return f"Updated group '{args['group_name']}'."


def _exec_activate_scene(args: dict) -> str:
    b = _get_bridge()
    group_id = _find_group_id(b, args["group_name"])
    if group_id is None:
        groups = [g["name"] for g in b.get_group().values()]
        return f"Unknown group '{args['group_name']}'. Available: {', '.join(groups)}"

    target = _normalize(args["scene_name"]).lower()
    for sid, scene in b.get_scene().items():
        if _normalize(scene.get("name", "")).lower() == target:
            b.activate_scene(group_id, sid)
            return f"Activated scene '{args['scene_name']}'."
    return f"Scene '{args['scene_name']}' not found."


def _exec_get_hue_status(args: dict) -> str:
    b = _get_bridge()
    lines = []
    for light in b.lights:
        state = "on" if light.on else "off"
        bri = light.brightness
        ct = getattr(light, "colortemp", None)
        ct_str = f", ~{round(1_000_000 / ct)}K" if ct else ""
        lines.append(f"{light.name} (id={light.light_id}): {state}, bri={bri}/254{ct_str}")
    return "\n".join(lines)


def _exec_save_user_info(args: dict) -> str:
    from datetime import datetime

    profile = _load_profile()
    profile[args["key"]] = {"value": args["value"], "updated": datetime.now().isoformat()}
    _save_profile(profile)
    return f"Saved: {args['key']}"


def _exec_pair_hue_bridge(args: dict) -> str:
    from phue import Bridge, PhueRegistrationException

    ip = args["bridge_ip"]
    try:
        b = Bridge(ip, config_file_path=str(CONFIG_DIR / ".python_hue"))
        username = b.username
        _save_config({"bridge_ip": ip, "username": username})
        lights = list(b.get_light_objects("name").keys())
        return f"Paired! Found {len(lights)} lights: {', '.join(lights)}"
    except PhueRegistrationException:
        return "Link button not pressed. Press it and try again within 30 seconds."
    except Exception as e:
        return f"Failed: {e}"


TOOL_DISPATCH = {
    "get_circadian_recommendation": _exec_get_circadian,
    "set_lights": _exec_set_lights,
    "set_group": _exec_set_group,
    "activate_scene": _exec_activate_scene,
    "get_hue_status": _exec_get_hue_status,
    "save_user_info": _exec_save_user_info,
    "pair_hue_bridge": _exec_pair_hue_bridge,
}


def execute_tool(name: str, args: dict) -> str:
    fn = TOOL_DISPATCH.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        return fn(args)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

from pathlib import Path

BASE_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text()


def _build_system_prompt() -> str:
    parts = [BASE_SYSTEM_PROMPT]
    profile = get_profile_context()
    if profile:
        parts.append(profile)
    lights = get_lights_context()
    if lights:
        parts.append(lights)
    return "\n\n".join(parts)


def _get_client() -> tuple[OpenAI, str]:
    base_url = os.getenv("FRESNEL_BASE_URL", "http://localhost:1234/v1")
    api_key = os.getenv("FRESNEL_API_KEY", "lm-studio")
    model = os.getenv("FRESNEL_MODEL", "qwen2.5-7b-instruct")
    return OpenAI(base_url=base_url, api_key=api_key), model


def chat(user_message: str, max_rounds: int = 5) -> str:
    """Send a message and handle the tool-call loop. Returns final text."""
    client, model = _get_client()
    system_prompt = _build_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
        )

        choice = response.choices[0]

        # If no tool calls, return the text response
        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            return choice.message.content or ""

        # Process tool calls
        messages.append(choice.message)
        for tc in choice.message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "Max tool rounds reached."
