"""Fast-path heuristics for common commands.

Pattern-matches user input and executes directly via phue — no LLM needed.
Returns None if the command doesn't match any known pattern.
"""

import re

from heylux.mcp.circadian import get_circadian_state
from heylux.mcp.hue import _get_bridge
from heylux.routines import run_routine, list_routines

BRIGHTNESS_STEP = 20  # percent per brighter/dimmer

# Sentinel values for ambient modes (daemon interprets these)
# Format: SENTINEL or SENTINEL:light_name or SENTINEL:light_name:fade_minutes
SHORTCUT_BREATHE_START = "__BREATHE_START__"
SHORTCUT_BREATHE_STOP = "__BREATHE_STOP__"
SHORTCUT_CANDLE_START = "__CANDLE_START__"


def _parse_duration(text: str) -> tuple[str, float]:
    """Extract a trailing duration like '10m', '30min', '5 minutes' from text.

    Returns (text_without_duration, minutes). Returns 0 if no duration found.
    """
    m = re.search(r'\b(\d+)\s*(?:m|min|mins|minutes?)\s*$', text)
    if m:
        return text[:m.start()].strip(), float(m.group(1))
    return text, 0


def _clean_voice_text(text: str) -> str:
    """Strip filler words that voice transcription adds but don't affect intent.

    Handles: "the", "my", "please", "for me", trailing punctuation,
    leading politeness ("can you", "could you please"), and inverted
    forms like "set the nightstand to candle".
    """
    # Strip trailing punctuation and filler phrases
    text = re.sub(r'[.,!?]+$', '', text).strip()
    text = re.sub(r'\s+(please|for me|for us)\s*$', '', text).strip()
    # Strip leading filler
    text = re.sub(r'^(can you |could you |please )+(set |turn |put |switch )?', '', text).strip()
    # "set X to candle" / "turn X to candle" → "candle on X"
    m = re.match(r'^(?:set|turn|put|switch)\s+(.+?)\s+to\s+(candle|candlelight|candle mode|breathe|breathing)$', text)
    if m:
        text = f"{m.group(2)} on {m.group(1)}"
    # Strip articles/possessives before light names (after "on")
    text = re.sub(r'\bon\s+(the|my|our)\s+', 'on ', text)
    return text


def try_shortcut(text: str) -> str | None:
    """Try to handle a command directly. Returns response text, or None to fall through to LLM."""
    text = text.strip().lower()
    text = _clean_voice_text(text)

    # --- Ambient modes (optionally targeting specific lights and duration) ---
    # Exact matches first, then parameterized
    # "candle 10m" / "candle mode 10m" (no light name, just duration)
    base, fade = _parse_duration(text)
    if base in ("candle", "candle mode", "candlelight") and fade:
        return f"{SHORTCUT_CANDLE_START}::{fade}"

    if text in ("candle", "candle mode", "candlelight"):
        return SHORTCUT_CANDLE_START

    # "candle on nightstand" / "candle on night stand 10m"
    for prefix in ("candle mode on ", "candlelight on ", "candle on "):
        if text.startswith(prefix):
            rest = text[len(prefix):].strip()
            rest, fade = _parse_duration(rest)
            if rest:
                return f"{SHORTCUT_CANDLE_START}:{rest}:{fade}"

    # "breathe" / "breathe on nightstand"
    for prefix in ("breathing mode on ", "breathing mode ", "breathe on ", "breathing on ", "breathe "):
        if text.startswith(prefix) and text != prefix.strip():
            light_name = text[len(prefix):].strip()
            return f"{SHORTCUT_BREATHE_START}:{light_name}"

    if text in ("breathe", "breathing", "breathing mode"):
        return SHORTCUT_BREATHE_START

    if text in ("stop", "stop breathing", "normal"):
        return SHORTCUT_BREATHE_STOP

    # --- On / Off ---
    if re.match(r"^(turn\s+)?(all\s+)?(my\s+)?(the\s+)?lights?\s+off$", text):
        return SHORTCUT_BREATHE_STOP

    if re.match(r"^(turn\s+)?(all\s+)?(my\s+)?(the\s+)?lights?\s+on$", text):
        return _all_on()

    if text in ("off", "goodnight", "good night"):
        return SHORTCUT_BREATHE_STOP

    if text == "on":
        return _all_on()

    # --- Brightness: absolute ---
    m = re.match(r"^(?:set\s+)?(?:brightness\s+(?:to\s+)?)?(\d{1,3})\s*%$", text)
    if m:
        return _set_brightness(int(m.group(1)))

    m = re.match(r"^dim\s+(?:to\s+)?(\d{1,3})\s*%?$", text)
    if m:
        return _set_brightness(int(m.group(1)))

    # --- Brightness: relative ---
    if text in ("brighter", "bright", "brighten", "more light", "more"):
        return _adjust_brightness(BRIGHTNESS_STEP)

    if text in ("dimmer", "dim", "darker", "less light", "less"):
        return _adjust_brightness(-BRIGHTNESS_STEP)

    m = re.match(r"^(brighter|dimmer)\s+(\d{1,3})\s*%?$", text)
    if m:
        step = int(m.group(2))
        return _adjust_brightness(step if m.group(1) == "brighter" else -step)

    # --- Circadian ---
    if text in ("circadian", "set for now", "set to now", "optimize", "auto"):
        return _apply_circadian()
    if re.search(r'\b(circadian|circadium|circanium)\b', text):
        return _apply_circadian()

    # --- Routines ---
    if text in ("routines", "list routines"):
        routines = list_routines()
        if not routines:
            return "No routines saved."
        lines = ["Routines:"]
        for name, desc in routines.items():
            lines.append(f"  {name} — {desc}")
        return "\n".join(lines)

    # Try matching a routine name directly, or with "mode" suffix
    result = run_routine(text)
    if result is not None:
        return result
    # "coding mode" → try "coding", "focus mode" → try "focus"
    if text.endswith(" mode"):
        result = run_routine(text[:-5])
        if result is not None:
            return result
    # "activate coding" → try "coding"
    for prefix in ("activate ", "set ", "switch to ", "turn on "):
        if text.startswith(prefix):
            result = run_routine(text[len(prefix):].rstrip(" mode"))
            if result is not None:
                return result

    return None


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _all_off() -> str:
    b = _get_bridge()
    for light in b.lights:
        b.set_light(light.light_id, {"on": False, "transitiontime": 10})
    return "All lights off."


def _all_on() -> str:
    b = _get_bridge()
    for light in b.lights:
        b.set_light(light.light_id, {"on": True, "transitiontime": 10})
    return "All lights on."


def _apply_circadian() -> str:
    b = _get_bridge()
    state = get_circadian_state()

    bri = round(state["brightness_pct"] * 254 / 100)
    ct = round(1_000_000 / state["kelvin"])

    cmd = {"on": True, "bri": bri, "ct": ct, "transitiontime": 20}

    active = [a.lower() for a in state["active_lights"]]

    for light in b.lights:
        if light.name.lower() in active:
            b.set_light(light.light_id, cmd)
        else:
            b.set_light(light.light_id, {"on": False, "transitiontime": 20})

    return (
        f"Circadian: {state['mode_name']} "
        f"({state['kelvin']}K, {state['brightness_pct']}%)"
    )


def _set_brightness(pct: int) -> str:
    pct = max(0, min(100, pct))
    b = _get_bridge()
    bri = round(pct * 254 / 100)
    for light in b.lights:
        if light.on:
            b.set_light(light.light_id, {"bri": bri, "transitiontime": 4})
    return f"Brightness: {pct}%"


def _adjust_brightness(step: int) -> str:
    b = _get_bridge()
    changed = []
    for light in b.lights:
        if light.on:
            current_pct = round(light.brightness * 100 / 254)
            new_pct = max(1, min(100, current_pct + step))
            new_bri = round(new_pct * 254 / 100)
            b.set_light(light.light_id, {"bri": new_bri, "transitiontime": 4})
            changed.append(new_pct)

    if not changed:
        return "No lights are on."

    avg = round(sum(changed) / len(changed))
    direction = "Brighter" if step > 0 else "Dimmer"
    return f"{direction}: ~{avg}%"
