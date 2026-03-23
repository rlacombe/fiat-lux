"""Fast-path heuristics for common commands.

Pattern-matches user input and executes directly via phue — no LLM needed.
Returns None if the command doesn't match any known pattern.
"""

import re

from fresnel.tools.circadian import get_circadian_state
from fresnel.tools.hue import _get_bridge, _find_group_id


def try_shortcut(text: str) -> str | None:
    """Try to handle a command directly. Returns response text, or None to fall through to LLM."""
    text = text.strip().lower()

    # Lights off
    if re.match(r"^(turn\s+)?(all\s+)?(my\s+)?(the\s+)?lights?\s+off$", text):
        return _all_off()

    # Lights on
    if re.match(r"^(turn\s+)?(all\s+)?(my\s+)?(the\s+)?lights?\s+on$", text):
        return _all_on()

    # Circadian / "set for now" / "optimize"
    if re.match(r"^(circadian|set\s+(for|to)\s+now|optimize)$", text):
        return _apply_circadian()

    # Brightness: "50%", "brightness 50", "dim to 30%"
    m = re.match(r"^(?:set\s+)?(?:brightness\s+(?:to\s+)?)?(\d{1,3})\s*%$", text)
    if m:
        return _set_brightness(int(m.group(1)))
    m = re.match(r"^dim\s+(?:to\s+)?(\d{1,3})\s*%?$", text)
    if m:
        return _set_brightness(int(m.group(1)))

    return None


def _all_off() -> str:
    b = _get_bridge()
    for light in b.lights:
        b.set_light(light.light_id, "on", False)
    return "All lights off."


def _all_on() -> str:
    b = _get_bridge()
    for light in b.lights:
        b.set_light(light.light_id, "on", True)
    return "All lights on."


def _apply_circadian() -> str:
    b = _get_bridge()
    state = get_circadian_state()

    bri = round(state["brightness_pct"] * 254 / 100)
    ct = round(1_000_000 / state["kelvin"])

    cmd = {"on": True, "bri": bri, "ct": ct, "transitiontime": 20}

    name_map = {light.name.lower(): light.light_id for light in b.lights}
    active = state["active_lights"]

    for light in b.lights:
        if light.name.lower() in [a.lower() for a in active]:
            b.set_light(light.light_id, cmd)
        else:
            b.set_light(light.light_id, "on", False)

    return (
        f"Circadian mode: {state['mode_name']} "
        f"({state['kelvin']}K, {state['brightness_pct']}%)"
    )


def _set_brightness(pct: int) -> str:
    pct = max(0, min(100, pct))
    b = _get_bridge()
    bri = round(pct * 254 / 100)
    for light in b.lights:
        if light.on:
            b.set_light(light.light_id, "bri", bri)
    return f"Brightness set to {pct}%."
