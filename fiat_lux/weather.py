"""Weather integration via Open-Meteo API.

Polls weather data periodically and caches it. Provides cloud cover,
sunrise/sunset, and weather conditions for the circadian engine and Lux.
Location is obtained via macOS CoreLocation (with user permission) or
manual entry, saved to ~/.config/fiat_lux/weather.json.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError

CONFIG_DIR = Path.home() / ".config" / "fiat_lux"
WEATHER_CONFIG = CONFIG_DIR / "weather.json"
WEATHER_CACHE = CONFIG_DIR / "weather_cache.json"

# Cache weather for 30 minutes
CACHE_MAX_AGE = 30 * 60

# Swift script to get location via CoreLocation
_LOCATION_SWIFT = r"""
import CoreLocation
import Foundation

class Locator: NSObject, CLLocationManagerDelegate {
    let mgr = CLLocationManager()
    let sema = DispatchSemaphore(value: 0)
    var result: CLLocation?

    override init() {
        super.init()
        mgr.delegate = self
    }

    func run() {
        mgr.requestWhenInUseAuthorization()
        mgr.requestLocation()
        _ = sema.wait(timeout: .now() + 10)
        if let loc = result {
            print("\(loc.coordinate.latitude),\(loc.coordinate.longitude)")
        } else {
            fputs("ERROR: Could not determine location\n", stderr)
            exit(1)
        }
    }

    func locationManager(_ m: CLLocationManager, didUpdateLocations locs: [CLLocation]) {
        result = locs.first
        sema.signal()
    }

    func locationManager(_ m: CLLocationManager, didFailWithError error: Error) {
        fputs("ERROR: \(error.localizedDescription)\n", stderr)
        sema.signal()
    }
}

Locator().run()
"""


def _load_config() -> dict[str, Any]:
    if WEATHER_CONFIG.exists():
        return json.loads(WEATHER_CONFIG.read_text())
    return {}


def _save_config(config: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    WEATHER_CONFIG.write_text(json.dumps(config, indent=2))


def get_location() -> tuple[float, float] | None:
    """Get saved lat/lon, or None if not configured."""
    config = _load_config()
    lat = config.get("latitude")
    lon = config.get("longitude")
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    return None


def get_location_description() -> str:
    """Get saved location as a human-readable string."""
    config = _load_config()
    city = config.get("city", "")
    lat = config.get("latitude")
    lon = config.get("longitude")
    if city:
        return city
    if lat is not None and lon is not None:
        return f"{lat:.2f}, {lon:.2f}"
    return ""


def _reverse_geocode_county(lat: float, lon: float) -> str:
    """Get the county/region name via Nominatim reverse geocoding."""
    try:
        url = (
            f"https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lon}&format=json&zoom=10"
        )
        req = Request(url, headers={"User-Agent": "fiat-lux/0.1"})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            addr = data.get("address", {})
            county = addr.get("county", "")
            # Strip "County", "Parish", etc. to get a searchable city name
            for suffix in (" County", " Parish", " Borough", " Municipality"):
                county = county.replace(suffix, "")
            return county.strip()
    except (URLError, json.JSONDecodeError, OSError):
        return ""


def _find_nearest_major_city(lat: float, lon: float, ip_city: str) -> str:
    """Find the most recognizable city near the given coordinates.

    Strategy: get the county name via reverse geocoding (e.g. "Los Angeles"),
    then search Open-Meteo's geocoding for both the county name and IP city.
    Pick the largest city within 50km by population.
    """
    import math
    from urllib.parse import quote

    candidates = []

    # Get county name and use it as a search query
    county = _reverse_geocode_county(lat, lon)

    queries = set()
    if ip_city:
        queries.add(ip_city)
    if county and county != ip_city:
        queries.add(county)
    for query in queries:
        try:
            url = (
                f"https://geocoding-api.open-meteo.com/v1/search"
                f"?name={quote(query)}&count=10"
            )
            req = Request(url, headers={"User-Agent": "fiat-lux/0.1"})
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                for r in data.get("results", []):
                    dlat = abs(r["latitude"] - lat)
                    dlon = abs(r["longitude"] - lon)
                    dist_km = math.sqrt(dlat**2 + dlon**2) * 111
                    pop = r.get("population", 0) or 0
                    if dist_km < 50:
                        name = r.get("name", "")
                        admin1 = r.get("admin1", "")
                        country = r.get("country", "")
                        candidates.append((pop, dist_km, name, admin1, country))
        except (URLError, json.JSONDecodeError, OSError):
            pass

    if candidates:
        # Pick the largest city by population
        candidates.sort(reverse=True)
        _, _, name, admin1, country = candidates[0]
        return ", ".join(filter(None, [name, admin1, country]))

    return ""


def request_ip_location() -> tuple[float, float, str] | None:
    """Get approximate location via IP geolocation. No permissions needed.

    Returns (lat, lon, city_description) or None.
    Uses IP for coordinates, then finds the nearest recognizable major city.
    """
    try:
        req = Request(
            "https://ipapi.co/json/",
            headers={"User-Agent": "fiat-lux/0.1"},
        )
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            lat = float(data["latitude"])
            lon = float(data["longitude"])
            ip_city = data.get("city", "")
            region = data.get("region", "")
            country = data.get("country_name", "")

            # Find the most recognizable nearby city
            major_city = _find_nearest_major_city(lat, lon, ip_city)
            if major_city:
                return lat, lon, major_city

            # Fall back to raw IP-reported city
            desc = ", ".join(filter(None, [ip_city, region, country]))
            return lat, lon, desc
    except (URLError, json.JSONDecodeError, KeyError, ValueError, OSError):
        pass
    return None


def request_macos_location() -> tuple[float, float] | None:
    """Request location via macOS CoreLocation. Returns (lat, lon) or None.

    Requires Location Services permission for the terminal app.
    Falls back to IP geolocation if CoreLocation fails.
    """
    if sys.platform != "darwin":
        return None

    try:
        result = subprocess.run(
            ["swift", "-e", _LOCATION_SWIFT],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and "," in result.stdout:
            parts = result.stdout.strip().split(",")
            return float(parts[0]), float(parts[1])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return None


def save_location(latitude: float, longitude: float, city: str = "") -> None:
    """Save location to config."""
    config = _load_config()
    config["latitude"] = latitude
    config["longitude"] = longitude
    if city:
        config["city"] = city
    config["updated"] = datetime.now().isoformat()
    _save_config(config)


def _load_cache() -> dict[str, Any] | None:
    """Load cached weather data if fresh enough."""
    if not WEATHER_CACHE.exists():
        return None
    try:
        cache = json.loads(WEATHER_CACHE.read_text())
        cached_at = cache.get("cached_at", 0)
        if time.time() - cached_at < CACHE_MAX_AGE:
            return cache
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _save_cache(data: dict[str, Any]) -> None:
    data["cached_at"] = time.time()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    WEATHER_CACHE.write_text(json.dumps(data, indent=2))


# WMO weather codes → human descriptions
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def fetch_weather(latitude: float, longitude: float) -> dict[str, Any] | None:
    """Fetch current weather from Open-Meteo. Returns parsed data or None on error."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&current=cloud_cover,weather_code,is_day,temperature_2m"
        f"&daily=sunrise,sunset,uv_index_max"
        f"&timezone=auto"
    )
    try:
        req = Request(url, headers={"User-Agent": "fiat-lux/0.1"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except (URLError, json.JSONDecodeError, OSError):
        return None


def get_weather() -> dict[str, Any] | None:
    """Get current weather data (from cache or fresh fetch).

    Returns a simplified dict with the fields Lux cares about, or None
    if weather isn't configured or the fetch fails.
    """
    # Check cache first
    cache = _load_cache()
    if cache and "weather" in cache:
        return cache["weather"]

    # Need location
    loc = get_location()
    if loc is None:
        return None

    lat, lon = loc
    raw = fetch_weather(lat, lon)
    if raw is None:
        return None

    # Parse into a clean structure
    current = raw.get("current", {})
    daily = raw.get("daily", {})

    weather_code = current.get("weather_code", 0)
    cloud_cover = current.get("cloud_cover", 0)

    # Today's sunrise/sunset
    sunrise = None
    sunset = None
    if daily.get("sunrise"):
        sunrise = daily["sunrise"][0]
    if daily.get("sunset"):
        sunset = daily["sunset"][0]

    result = {
        "cloud_cover": cloud_cover,
        "weather_code": weather_code,
        "weather_description": WMO_CODES.get(weather_code, f"Code {weather_code}"),
        "is_day": bool(current.get("is_day", True)),
        "temperature_c": current.get("temperature_2m"),
        "sunrise": sunrise,
        "sunset": sunset,
        "uv_index_max": daily.get("uv_index_max", [None])[0],
    }

    _save_cache({"weather": result})
    return result


def get_weather_context() -> str:
    """Return weather summary for injection into the system prompt."""
    w = get_weather()
    if w is None:
        return ""

    lines = ["## Current Weather"]
    city = get_location_description()
    if city:
        lines.append(f"- Location: {city}")
    lines.append(f"- Conditions: {w['weather_description']}, {w['cloud_cover']}% cloud cover")
    if w["temperature_c"] is not None:
        lines.append(f"- Temperature: {w['temperature_c']}°C")
    if w["sunrise"]:
        # Extract just the time part
        try:
            sr = datetime.fromisoformat(w["sunrise"]).strftime("%H:%M")
            ss = datetime.fromisoformat(w["sunset"]).strftime("%H:%M")
            lines.append(f"- Sunrise: {sr}, Sunset: {ss}")
        except ValueError:
            pass
    if w["uv_index_max"] is not None:
        lines.append(f"- UV index (today's max): {w['uv_index_max']}")

    return "\n".join(lines)


def get_brightness_adjustment() -> float:
    """Return a brightness multiplier based on cloud cover.

    1.0 = no adjustment (clear sky or no weather data)
    >1.0 = boost brightness (overcast/dark conditions)

    Used by the circadian engine to compensate for lack of natural light.
    """
    w = get_weather()
    if w is None:
        return 1.0

    cloud = w.get("cloud_cover", 0)

    # Scale: 0% clouds → 1.0x, 100% clouds → 1.3x (30% brightness boost)
    return 1.0 + (cloud / 100) * 0.3


def get_actual_sunrise_sunset() -> tuple[float, float] | None:
    """Return today's (sunrise_hour, sunset_hour) as floats, or None.

    e.g. (7.25, 19.5) for 7:15am sunrise, 7:30pm sunset.
    Used by the circadian engine to shift waypoints to match actual daylight.
    """
    w = get_weather()
    if w is None:
        return None

    try:
        sr = datetime.fromisoformat(w["sunrise"])
        ss = datetime.fromisoformat(w["sunset"])
        sr_hour = sr.hour + sr.minute / 60
        ss_hour = ss.hour + ss.minute / 60
        return sr_hour, ss_hour
    except (ValueError, TypeError, KeyError):
        return None
