"""MCP tools for weather integration — setup and querying."""

from typing import Any

from claude_agent_sdk import tool

from fiat_lux.weather import (
    get_location,
    get_location_description,
    get_weather,
    request_ip_location,
    request_macos_location,
    save_location,
)


def _text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _error(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


@tool(
    "setup_weather",
    "Set up weather integration for adaptive lighting. Three options: "
    "(1) auto — uses IP geolocation for city-level accuracy, no permissions needed, "
    "works instantly. Best default. "
    "(2) use_macos_location — precise GPS via CoreLocation, requires permission. "
    "(3) manual lat/lon — user provides coordinates. "
    "Always confirm with the user before proceeding. Weather data lets Lux adjust "
    "brightness on cloudy days and track actual sunrise/sunset times.",
    {
        "type": "object",
        "properties": {
            "auto": {
                "type": "boolean",
                "description": (
                    "If true, use IP geolocation for city-level location. "
                    "No permissions needed. Accurate enough for weather/sunrise."
                ),
            },
            "use_macos_location": {
                "type": "boolean",
                "description": (
                    "If true, request precise GPS via macOS CoreLocation. "
                    "Only use after the user gives explicit permission."
                ),
            },
            "latitude": {
                "type": "number",
                "description": "Manual latitude.",
            },
            "longitude": {
                "type": "number",
                "description": "Manual longitude.",
            },
        },
        "required": [],
    },
)
async def setup_weather(args: dict[str, Any]) -> dict[str, Any]:
    # Check current state
    loc = get_location()
    if loc and not args:
        return _text(
            f"Weather is already configured (lat={loc[0]:.4f}, lon={loc[1]:.4f}).\n"
            "To update, call again with auto=true or manual coordinates."
        )

    # Option 1: IP geolocation (default, no permissions)
    if args.get("auto", False):
        result = request_ip_location()
        if result is None:
            return _error(
                "IP geolocation failed (network issue?). "
                "Ask the user for their city and use manual coordinates instead."
            )
        lat, lon, city = result
        save_location(lat, lon, city)
        w = get_weather()
        city_str = f" ({city})" if city else ""
        if w:
            return _text(
                f"Location detected{city_str} — {lat:.2f}, {lon:.2f}\n"
                f"Current weather: {w['weather_description']}, {w['cloud_cover']}% cloud cover.\n"
                f"Sunrise: {w.get('sunrise', '?')}, Sunset: {w.get('sunset', '?')}\n\n"
                "Weather-adaptive lighting is now active."
            )
        return _text(f"Location saved{city_str}. Weather fetch will start on next poll.")

    # Option 2: macOS CoreLocation (precise, needs permission)
    if args.get("use_macos_location"):
        result = request_macos_location()
        if result is None:
            # Fall back to IP geolocation
            ip_result = request_ip_location()
            if ip_result:
                lat, lon, city = ip_result
                save_location(lat, lon, city)
                w = get_weather()
                city_str = f" ({city})" if city else ""
                if w:
                    return _text(
                        f"macOS location unavailable — used IP geolocation instead{city_str}.\n"
                        f"Location: {lat:.2f}, {lon:.2f}\n"
                        f"Current weather: {w['weather_description']}, {w['cloud_cover']}% cloud cover.\n"
                        f"Sunrise: {w.get('sunrise', '?')}, Sunset: {w.get('sunset', '?')}\n\n"
                        "Weather-adaptive lighting is now active."
                    )
            return _error(
                "Could not get location. Ask the user for their city "
                "and use manual coordinates."
            )
        lat, lon = result
        save_location(lat, lon)
        w = get_weather()
        if w:
            return _text(
                f"Location set ({lat:.4f}, {lon:.4f}).\n"
                f"Current weather: {w['weather_description']}, {w['cloud_cover']}% cloud cover.\n"
                f"Sunrise: {w.get('sunrise', '?')}, Sunset: {w.get('sunset', '?')}\n\n"
                "Weather-adaptive lighting is now active."
            )
        return _text(f"Location saved ({lat:.4f}, {lon:.4f}). Weather fetch will start on next poll.")

    # Option 3: Manual coordinates
    lat = args.get("latitude")
    lon = args.get("longitude")
    if lat is not None and lon is not None:
        save_location(lat, lon)
        w = get_weather()
        if w:
            return _text(
                f"Location set ({lat:.4f}, {lon:.4f}).\n"
                f"Current weather: {w['weather_description']}, {w['cloud_cover']}% cloud cover.\n"
                f"Sunrise: {w.get('sunrise', '?')}, Sunset: {w.get('sunset', '?')}\n\n"
                "Weather-adaptive lighting is now active."
            )
        return _text(f"Location saved ({lat}, {lon}). Weather fetch will start on next poll.")

    return _error(
        "Provide auto=true (recommended, no permissions), "
        "use_macos_location=true (precise GPS), "
        "or manual latitude/longitude."
    )


@tool(
    "get_current_weather",
    "Get the current weather conditions for lighting decisions. Returns cloud cover, "
    "weather description, sunrise/sunset times, and UV index. Uses cached data "
    "(refreshed every 30 minutes). Requires weather to be set up first.",
    {},
)
async def get_current_weather(args: dict[str, Any]) -> dict[str, Any]:
    w = get_weather()
    if w is None:
        return _error(
            "Weather not configured. Use setup_weather first to set your location."
        )

    location = get_location_description()
    header = f"**{w['weather_description']}** — {w['cloud_cover']}% cloud cover"
    if location:
        header = f"**{location}**: {w['weather_description']}, {w['cloud_cover']}% cloud cover"
    lines = [header]
    if w.get("temperature_c") is not None:
        lines.append(f"Temperature: {w['temperature_c']}°C")
    if w.get("sunrise"):
        lines.append(f"Sunrise: {w['sunrise']}")
    if w.get("sunset"):
        lines.append(f"Sunset: {w['sunset']}")
    if w.get("uv_index_max") is not None:
        lines.append(f"UV index (today's max): {w['uv_index_max']}")

    # Brightness recommendation
    cloud = w.get("cloud_cover", 0)
    if cloud > 70:
        lines.append(f"\n**Lighting note:** Heavy cloud cover — indoor brightness should be boosted ~{round(cloud * 0.3)}% to compensate for low natural light.")
    elif cloud > 40:
        lines.append(f"\n**Lighting note:** Moderate cloud cover — slight brightness boost recommended.")

    return _text("\n".join(lines))


@tool(
    "update_location",
    "Update the saved location for weather data. Use this when the user is traveling "
    "or has moved. Use auto=true (IP geolocation, no permissions) or manual coordinates.",
    {
        "type": "object",
        "properties": {
            "auto": {
                "type": "boolean",
                "description": "If true, detect location via IP geolocation.",
            },
            "latitude": {"type": "number"},
            "longitude": {"type": "number"},
        },
        "required": [],
    },
)
async def update_location(args: dict[str, Any]) -> dict[str, Any]:
    # Delegate to setup_weather — same logic
    return await setup_weather(args)


ALL_WEATHER_TOOLS = [
    setup_weather,
    get_current_weather,
    update_location,
]
