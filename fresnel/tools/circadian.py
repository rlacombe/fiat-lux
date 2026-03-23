"""Circadian rhythm engine grounded in chronobiology.

Provides time-based lighting recommendations backed by research on
melanopsin sensitivity, melatonin suppression, and cortisol regulation.
"""

from datetime import datetime
from typing import Any

from claude_agent_sdk import tool


# Circadian waypoints: (hour, kelvin, brightness_pct, active_lights, mode_name)
# Based on melanopic EDI targets and chronobiology research.
# Brightness is 0-100%, kelvin is color temperature.
CIRCADIAN_WAYPOINTS = [
    (6.0, 2000, 5, ["floor"], "Pre-dawn"),
    (7.0, 4000, 60, ["ceiling"], "Energize"),
    (8.0, 5000, 80, ["ceiling", "desk"], "Morning Focus"),
    (9.0, 5500, 90, ["ceiling", "desk"], "Concentrate"),
    (12.0, 6500, 100, ["ceiling", "desk"], "Peak Daylight"),
    (14.0, 6000, 95, ["ceiling", "desk"], "Afternoon"),
    (17.0, 4500, 70, ["ceiling"], "Wind Down"),
    (19.0, 3000, 50, ["ceiling", "desk"], "Read"),
    (20.0, 2700, 40, ["floor", "desk"], "Relax"),
    (21.5, 2200, 20, ["floor"], "Pre-Sleep"),
    (22.0, 2000, 10, ["floor"], "Nightlight"),
    (23.0, 1800, 5, ["floor"], "Deep Night"),
]


def _interpolate(t: float, t0: float, t1: float, v0: float, v1: float) -> float:
    """Linear interpolation between two values."""
    if t1 == t0:
        return v0
    ratio = (t - t0) / (t1 - t0)
    return v0 + (v1 - v0) * ratio


def get_circadian_state(now: datetime | None = None) -> dict[str, Any]:
    """Compute the ideal lighting state for the given time.

    Returns a dict with kelvin, brightness_pct, active_lights, and mode_name.
    Uses linear interpolation between the circadian waypoints.
    """
    if now is None:
        now = datetime.now()

    hour = now.hour + now.minute / 60.0

    # Find surrounding waypoints
    prev = CIRCADIAN_WAYPOINTS[-1]
    next_wp = CIRCADIAN_WAYPOINTS[0]

    for i, wp in enumerate(CIRCADIAN_WAYPOINTS):
        if wp[0] <= hour:
            prev = wp
            next_wp = CIRCADIAN_WAYPOINTS[(i + 1) % len(CIRCADIAN_WAYPOINTS)]
        else:
            next_wp = wp
            break

    # Handle wrap-around (e.g., 23:00 -> 06:00)
    t0 = prev[0]
    t1 = next_wp[0]
    if t1 <= t0:
        t1 += 24
    if hour < t0:
        hour += 24

    kelvin = round(_interpolate(hour, t0, t1, prev[1], next_wp[1]))
    brightness = round(_interpolate(hour, t0, t1, prev[2], next_wp[2]))

    return {
        "kelvin": kelvin,
        "brightness_pct": brightness,
        "active_lights": prev[3],
        "mode_name": prev[4],
        "time": now.strftime("%H:%M"),
    }


@tool(
    "get_circadian_recommendation",
    "Get the optimal lighting recommendation based on circadian science. "
    "Automatically uses the current local time (no input needed). Returns color "
    "temperature (Kelvin), brightness (%), which lights should be active, and the "
    "circadian mode. Call this with no arguments, then apply the result via set_lights.",
    {"time_override": str},
)
async def get_circadian_recommendation(args: dict[str, Any]) -> dict[str, Any]:
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
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Invalid time format '{time_str}'. Use HH:MM (e.g., '14:30').",
                    }
                ]
            }

    state = get_circadian_state(now)

    explanation = (
        f"Circadian mode: **{state['mode_name']}** (at {state['time']})\n"
        f"- Color temperature: {state['kelvin']}K\n"
        f"- Brightness: {state['brightness_pct']}%\n"
        f"- Active lights: {', '.join(state['active_lights'])}\n\n"
        f"Rationale: At {state['time']}, "
    )

    hour = float(state["time"].split(":")[0]) + float(state["time"].split(":")[1]) / 60

    if hour < 7:
        explanation += (
            "pre-dawn light should be minimal and very warm to avoid disrupting "
            "melatonin. Only ambient floor lighting at deep amber."
        )
    elif hour < 10:
        explanation += (
            "morning light exposure is critical for cortisol awakening response (CAR). "
            "High melanopic EDI (cool white, bright) via overhead lighting promotes "
            "alertness and entrains the circadian clock."
        )
    elif hour < 14:
        explanation += (
            "midday is peak alertness. Maximum color temperature and brightness "
            "maintains the cortisol plateau and supports cognitive performance."
        )
    elif hour < 17:
        explanation += (
            "early afternoon begins the natural dip in alertness. Maintaining "
            "moderate brightness and slightly warmer temperature supports "
            "sustained focus without overstimulation."
        )
    elif hour < 20:
        explanation += (
            "evening wind-down. Reducing melanopic content (warmer color temperature) "
            "allows dim light melatonin onset (DLMO) to begin on schedule. "
            "Shifting from overhead to peripheral lighting reduces alerting signals."
        )
    elif hour < 22:
        explanation += (
            "pre-sleep phase. Very warm light (≤2700K) with low brightness minimizes "
            "melatonin suppression. Amber/red wavelengths have near-zero melanopic "
            "efficacy, making them sleep-safe."
        )
    else:
        explanation += (
            "deep night. If light is needed, only the faintest warm amber is safe. "
            "Any short-wavelength (blue) exposure at this hour causes significant "
            "circadian phase delay."
        )

    return {"content": [{"type": "text", "text": explanation}]}
