"""Tests for the weather integration."""

import json
import time
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from fiat_lux.weather import (
    WMO_CODES,
    _load_cache,
    _save_cache,
    fetch_weather,
    get_actual_sunrise_sunset,
    get_brightness_adjustment,
    get_location,
    get_weather,
    get_weather_context,
    save_location,
)


@pytest.fixture
def weather_files(tmp_path, monkeypatch):
    """Redirect weather files to tmp dir."""
    config = tmp_path / "weather.json"
    cache = tmp_path / "weather_cache.json"
    monkeypatch.setattr("fiat_lux.weather.WEATHER_CONFIG", config)
    monkeypatch.setattr("fiat_lux.weather.WEATHER_CACHE", cache)
    monkeypatch.setattr("fiat_lux.weather.CONFIG_DIR", tmp_path)
    return config, cache


SAMPLE_WEATHER = {
    "cloud_cover": 75,
    "weather_code": 3,
    "weather_description": "Overcast",
    "is_day": True,
    "temperature_c": 12.5,
    "sunrise": "2026-03-23T07:15:00",
    "sunset": "2026-03-23T19:30:00",
    "uv_index_max": 4.2,
}


class TestLocation:
    def test_save_and_get(self, weather_files):
        save_location(48.8566, 2.3522)
        loc = get_location()
        assert loc is not None
        assert abs(loc[0] - 48.8566) < 0.001
        assert abs(loc[1] - 2.3522) < 0.001

    def test_no_location(self, weather_files):
        assert get_location() is None


class TestCache:
    def test_fresh_cache(self, weather_files):
        _save_cache({"weather": SAMPLE_WEATHER})
        cached = _load_cache()
        assert cached is not None
        assert cached["weather"]["cloud_cover"] == 75

    def test_stale_cache(self, weather_files, monkeypatch):
        _save_cache({"weather": SAMPLE_WEATHER})
        # Make cache appear old
        _, cache_file = weather_files
        data = json.loads(cache_file.read_text())
        data["cached_at"] = time.time() - 3600  # 1 hour old
        cache_file.write_text(json.dumps(data))
        assert _load_cache() is None

    def test_no_cache(self, weather_files):
        assert _load_cache() is None


class TestGetWeather:
    def test_returns_cached(self, weather_files):
        _save_cache({"weather": SAMPLE_WEATHER})
        w = get_weather()
        assert w is not None
        assert w["cloud_cover"] == 75

    def test_returns_none_without_location(self, weather_files):
        assert get_weather() is None

    @patch("fiat_lux.weather.fetch_weather")
    def test_fetches_when_no_cache(self, mock_fetch, weather_files):
        save_location(48.8566, 2.3522)
        mock_fetch.return_value = {
            "current": {
                "cloud_cover": 40,
                "weather_code": 2,
                "is_day": 1,
                "temperature_2m": 18.0,
            },
            "daily": {
                "sunrise": ["2026-03-23T07:00:00"],
                "sunset": ["2026-03-23T19:45:00"],
                "uv_index_max": [5.0],
            },
        }
        w = get_weather()
        assert w is not None
        assert w["cloud_cover"] == 40
        assert w["weather_description"] == "Partly cloudy"


class TestBrightnessAdjustment:
    def test_clear_sky(self, weather_files):
        _save_cache({"weather": {**SAMPLE_WEATHER, "cloud_cover": 0}})
        assert get_brightness_adjustment() == 1.0

    def test_overcast(self, weather_files):
        _save_cache({"weather": {**SAMPLE_WEATHER, "cloud_cover": 100}})
        adj = get_brightness_adjustment()
        assert adj == pytest.approx(1.3, abs=0.01)

    def test_partial_clouds(self, weather_files):
        _save_cache({"weather": {**SAMPLE_WEATHER, "cloud_cover": 50}})
        adj = get_brightness_adjustment()
        assert 1.0 < adj < 1.3

    def test_no_weather(self, weather_files):
        assert get_brightness_adjustment() == 1.0


class TestSunriseSunset:
    def test_returns_hours(self, weather_files):
        _save_cache({"weather": SAMPLE_WEATHER})
        result = get_actual_sunrise_sunset()
        assert result is not None
        sr, ss = result
        assert abs(sr - 7.25) < 0.01  # 7:15
        assert abs(ss - 19.5) < 0.01  # 19:30

    def test_no_weather(self, weather_files):
        assert get_actual_sunrise_sunset() is None


class TestWeatherContext:
    def test_generates_context(self, weather_files):
        _save_cache({"weather": SAMPLE_WEATHER})
        ctx = get_weather_context()
        assert "## Current Weather" in ctx
        assert "Overcast" in ctx
        assert "75%" in ctx

    def test_empty_without_weather(self, weather_files):
        assert get_weather_context() == ""


class TestWMOCodes:
    def test_known_codes(self):
        assert WMO_CODES[0] == "Clear sky"
        assert WMO_CODES[3] == "Overcast"
        assert WMO_CODES[95] == "Thunderstorm"
