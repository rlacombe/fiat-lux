"""Light pulse effects for notifications and ambient breathing.

Uses phue directly for low-latency light control.
"""

import asyncio
import random
import time
from typing import Any

from fiat_lux.tools.hue import _get_bridge


def _save_light_state(bridge, light_id: int) -> dict:
    """Capture current light state for later restore."""
    state = bridge.get_light(light_id)["state"]
    saved = {
        "on": state["on"],
        "bri": state.get("bri", 254),
        "colormode": state.get("colormode", "ct"),
    }
    if "ct" in state:
        saved["ct"] = state["ct"]
    if "hue" in state:
        saved["hue"] = state["hue"]
        saved["sat"] = state["sat"]
    return saved


def _restore_light_state(bridge, light_id: int, saved: dict) -> None:
    """Restore a previously saved light state."""
    cmd = {"transitiontime": 25}  # 2.5s fade back
    if saved["colormode"] == "ct":
        cmd["ct"] = saved.get("ct", 370)
    else:
        cmd["hue"] = saved.get("hue", 0)
        cmd["sat"] = saved.get("sat", 0)
    cmd["bri"] = saved["bri"]
    cmd["on"] = saved["on"]
    bridge.set_light(light_id, cmd)


def breathing_pulse(
    light_name: str = "Desk lamp",
    hue: int = 46920,
    saturation: int = 160,
    breaths: int = 2,
) -> None:
    """Perform a breathing pulse on a single light.

    Args:
        light_name: Name of the light to pulse.
        hue: Hue value (46920=blue, 8000=amber).
        saturation: Color saturation (0-254).
        breaths: Number of breath cycles.
    """
    b = _get_bridge()

    # Find light by name
    light_id = None
    for light in b.lights:
        if light.name.lower() == light_name.lower():
            light_id = light.light_id
            break
    if light_id is None:
        return

    saved = _save_light_state(b, light_id)

    try:
        # Initial inhale — fade into color
        b.set_light(light_id, {
            "on": True,
            "hue": hue,
            "sat": saturation,
            "bri": 254,
            "transitiontime": 20,  # 2s inhale
        })
        time.sleep(2.2)

        for _ in range(breaths - 1):
            # Exhale
            b.set_light(light_id, {
                "bri": 40,
                "transitiontime": 30,  # 3s exhale
            })
            time.sleep(3.2)

            # Inhale
            b.set_light(light_id, {
                "bri": 254,
                "transitiontime": 20,  # 2s inhale
            })
            time.sleep(2.2)

        # Final exhale
        b.set_light(light_id, {
            "bri": 40,
            "transitiontime": 30,
        })
        time.sleep(3.2)

    finally:
        _restore_light_state(b, light_id, saved)


# Preset pulses for calendar alerts
AMBER_HUE = 8000
BLUE_HUE = 46920


def pulse_heads_up(light_name: str = "Desk lamp") -> None:
    """Amber breathing pulse — meeting in 5 minutes."""
    breathing_pulse(light_name, hue=AMBER_HUE, saturation=200, breaths=2)


def pulse_starting_now(light_name: str = "Desk lamp") -> None:
    """Blue breathing pulse — meeting in 15 seconds."""
    breathing_pulse(light_name, hue=BLUE_HUE, saturation=160, breaths=2)


# ---------------------------------------------------------------------------
# Continuous breathing mode
# ---------------------------------------------------------------------------

BREATHE_INHALE = 4.0  # seconds to fade up
BREATHE_EXHALE = 6.0  # seconds to fade down
BREATHE_BRI_HIGH = 150  # ~60% brightness
BREATHE_BRI_LOW = 25  # ~10% brightness
# Simulate ~1500K via hue/sat (below the bridge's 2000K CT floor)
BREATHE_HUE = 7500  # deeper amber, more golden
BREATHE_SAT = 245


def _save_all_states(bridge) -> dict[int, dict]:
    """Capture state of all lights for later restore."""
    saved = {}
    for light in bridge.lights:
        saved[light.light_id] = _save_light_state(bridge, light.light_id)
    return saved


def _restore_all_states(bridge, saved: dict[int, dict]) -> None:
    """Restore all lights to their saved states."""
    for light_id, state in saved.items():
        _restore_light_state(bridge, light_id, state)


def _breathe_tick(bridge, light_ids: list[int], inhale: bool) -> None:
    """Execute a single inhale or exhale on all lights."""
    bri = BREATHE_BRI_HIGH if inhale else BREATHE_BRI_LOW
    duration = BREATHE_INHALE if inhale else BREATHE_EXHALE
    cmd = {
        "on": True,
        "bri": bri,
        "hue": BREATHE_HUE,
        "sat": BREATHE_SAT,
        "transitiontime": round(duration * 10),
    }
    for lid in light_ids:
        bridge.set_light(lid, cmd)


async def breathing_mode_loop(light_ids: list[int] | None = None) -> dict[int, Any]:
    """Run continuous breathing on lights until cancelled.

    Returns the saved states dict so the caller can restore after cancellation.
    This is an async coroutine meant to run as an asyncio.Task.
    """
    b = _get_bridge()

    if light_ids is None:
        light_ids = [l.light_id for l in b.lights]

    saved = _save_all_states(b)

    # Initial setup: set all lights to deep amber at low brightness
    for lid in light_ids:
        b.set_light(lid, {
            "on": True,
            "bri": BREATHE_BRI_LOW,
            "hue": BREATHE_HUE,
            "sat": BREATHE_SAT,
            "transitiontime": 20,
        })
    await asyncio.sleep(2.0)

    try:
        while True:
            # Inhale
            await asyncio.to_thread(_breathe_tick, b, light_ids, True)
            await asyncio.sleep(BREATHE_INHALE + 0.2)
            # Exhale
            await asyncio.to_thread(_breathe_tick, b, light_ids, False)
            await asyncio.sleep(BREATHE_EXHALE + 0.2)
    except asyncio.CancelledError:
        # Restore lights to pre-breathing state
        await asyncio.to_thread(_restore_all_states, b, saved)
        raise


# ---------------------------------------------------------------------------
# Candle mode — flickering candlelight simulation
# ---------------------------------------------------------------------------

# Candle: deep red-amber, biased towards red end of flame
CANDLE_HUE_CENTER = 2500  # deep red-amber
CANDLE_HUE_DRIFT = 2000  # wanders 500-4500 (red to amber)
CANDLE_SAT = 250  # near-max saturation

# Breathing baseline — the slow underlying swell
CANDLE_INHALE = 4.0  # seconds for slow swell up
CANDLE_EXHALE = 6.0  # seconds for slow fade down
CANDLE_BRI_HIGH = 140  # peak of breath
CANDLE_BRI_LOW = 55  # trough of breath (narrower range = less dramatic)

# Flicker: gentle jitter layered on top of the breathing
CANDLE_FLICKER_RANGE = 30  # +/- this from the current breath position
CANDLE_TICK = (1.0, 2.0)  # unhurried pace

# Gust: periodic wind dip — softer
CANDLE_GUST_BRI = 20  # dims but doesn't snuff out
CANDLE_GUST_INTERVAL = (12.0, 25.0)  # seconds between gusts
CANDLE_GUST_DOWN = 1.5  # gentler dip
CANDLE_GUST_UP = 4.0  # slow recovery


def _candle_tick(bridge, light_ids: list[int], breath_bri: int) -> None:
    """One flicker tick — each light jitters around the current breath level."""
    for lid in light_ids:
        # Random jitter around the breathing baseline
        bri = breath_bri + random.randint(-CANDLE_FLICKER_RANGE, CANDLE_FLICKER_RANGE)
        bri = max(1, min(254, bri))
        # Color drifts per light
        hue = CANDLE_HUE_CENTER + random.randint(-CANDLE_HUE_DRIFT, CANDLE_HUE_DRIFT)
        sat = CANDLE_SAT + random.randint(-15, 5)
        # Long smooth transitions (8-15 deciseconds = 0.8-1.5s) so changes are gentle
        tt = random.randint(8, 15)
        bridge.set_light(lid, {"bri": bri, "hue": hue, "sat": sat, "transitiontime": tt})


def _candle_gust(bridge, light_ids: list[int]) -> None:
    """Wind gust — all lights dim together."""
    gust_bri = random.randint(CANDLE_GUST_BRI, 20)
    for lid in light_ids:
        bridge.set_light(lid, {
            "bri": gust_bri,
            "transitiontime": round(CANDLE_GUST_DOWN * 10),
        })


def _candle_recover(bridge, light_ids: list[int], breath_bri: int) -> None:
    """Recover from a gust back to current breath level."""
    for lid in light_ids:
        bri = breath_bri + random.randint(-20, 20)
        bri = max(1, min(254, bri))
        bridge.set_light(lid, {
            "bri": bri,
            "hue": CANDLE_HUE_CENTER,
            "sat": CANDLE_SAT,
            "transitiontime": round(CANDLE_GUST_UP * 10),
        })


async def candle_mode_loop(
    light_ids: list[int] | None = None,
    fade_out_minutes: float = 0,
) -> None:
    """Run continuous candle mode on lights until cancelled.

    Combines breathing (slow underlying swell) with flame-like randomness
    (per-light flicker jitter and periodic wind gusts).

    If fade_out_minutes > 0, gradually dims over that duration then turns off.
    """
    b = _get_bridge()

    if light_ids is None:
        light_ids = [l.light_id for l in b.lights]

    saved = _save_all_states(b)

    # Fade into candle color
    for lid in light_ids:
        b.set_light(lid, {
            "on": True,
            "hue": CANDLE_HUE_CENTER,
            "sat": CANDLE_SAT,
            "bri": CANDLE_BRI_LOW,
            "transitiontime": 20,
        })
    await asyncio.sleep(2.2)

    try:
        import math
        cycle = CANDLE_INHALE + CANDLE_EXHALE
        next_gust = asyncio.get_event_loop().time() + random.uniform(*CANDLE_GUST_INTERVAL)
        start_time = asyncio.get_event_loop().time()
        fade_out_seconds = fade_out_minutes * 60

        while True:
            now = asyncio.get_event_loop().time()
            elapsed = now - start_time

            # Fade-out: gradually reduce the envelope over time
            if fade_out_seconds > 0:
                fade = max(0.0, 1.0 - elapsed / fade_out_seconds)
                if fade <= 0:
                    # Time's up — turn off and exit
                    for lid in light_ids:
                        b.set_light(lid, {"on": False, "transitiontime": 30})
                    return
            else:
                fade = 1.0

            # Compute breathing baseline: sinusoidal swell, scaled by fade
            phase = ((now - start_time) % cycle) / cycle
            wave = (math.sin(phase * 2 * math.pi - math.pi / 2) + 1) / 2
            bri_low = round(CANDLE_BRI_LOW * fade)
            bri_high = round(CANDLE_BRI_HIGH * fade)
            breath_bri = max(1, round(bri_low + wave * (bri_high - bri_low)))

            # Check for gust
            if now >= next_gust:
                await asyncio.to_thread(_candle_gust, b, light_ids)
                await asyncio.sleep(CANDLE_GUST_DOWN + 0.3)
                await asyncio.to_thread(_candle_recover, b, light_ids, breath_bri)
                await asyncio.sleep(CANDLE_GUST_UP + 0.3)
                next_gust = asyncio.get_event_loop().time() + random.uniform(*CANDLE_GUST_INTERVAL)
            else:
                # Flicker tick with jitter around the breathing baseline
                await asyncio.to_thread(_candle_tick, b, light_ids, breath_bri)
                await asyncio.sleep(random.uniform(*CANDLE_TICK))

    except asyncio.CancelledError:
        await asyncio.to_thread(_restore_all_states, b, saved)
        raise
