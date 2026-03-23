"""Tests for the circadian rhythm engine."""

from datetime import datetime
from unittest.mock import patch

import pytest

from fiat_lux.mcp.circadian import get_circadian_state


@pytest.fixture(autouse=True)
def no_weather():
    """Disable weather integration so tests use the static circadian curve."""
    with patch("fiat_lux.weather.get_actual_sunrise_sunset", return_value=None), \
         patch("fiat_lux.weather.get_brightness_adjustment", return_value=1.0):
        yield


def _at(hour: int, minute: int = 0) -> datetime:
    """Create a datetime at a specific time today."""
    return datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)


class TestKnownWaypoints:
    def test_6am_predawn(self):
        state = get_circadian_state(_at(6, 0))
        assert state["mode_name"] == "Pre-dawn"
        assert state["kelvin"] == 2000
        assert state["brightness_pct"] == 5

    def test_noon_peak(self):
        state = get_circadian_state(_at(12, 0))
        assert state["mode_name"] == "Peak Daylight"
        assert state["kelvin"] == 6500
        assert state["brightness_pct"] == 100

    def test_9am_concentrate(self):
        state = get_circadian_state(_at(9, 0))
        assert state["mode_name"] == "Concentrate"
        assert state["kelvin"] == 5500
        assert state["brightness_pct"] == 90

    def test_8pm_relax(self):
        state = get_circadian_state(_at(20, 0))
        assert state["mode_name"] == "Relax"
        assert state["kelvin"] == 2700
        assert state["brightness_pct"] == 40


class TestInterpolation:
    def test_7_30_between_energize_and_morning(self):
        state = get_circadian_state(_at(7, 30))
        # Between 7:00 (4000K, 60%) and 8:00 (5000K, 80%)
        assert 4000 < state["kelvin"] < 5000
        assert 60 < state["brightness_pct"] < 80
        assert state["mode_name"] == "Energize"

    def test_10_30_between_concentrate_and_peak(self):
        state = get_circadian_state(_at(10, 30))
        # Between 9:00 (5500K, 90%) and 12:00 (6500K, 100%)
        assert 5500 < state["kelvin"] < 6500
        assert 90 < state["brightness_pct"] < 100

    def test_interpolation_is_linear(self):
        # At exact midpoint between 7:00 and 8:00
        state = get_circadian_state(_at(7, 30))
        assert state["kelvin"] == 4500  # midpoint of 4000-5000
        assert state["brightness_pct"] == 70  # midpoint of 60-80


class TestMidnightWrap:
    def test_1am_interpolates(self):
        state = get_circadian_state(_at(1, 0))
        # Between 23:00 (1800K, 5%) and 6:00 (2000K, 5%)
        assert state["kelvin"] >= 1800
        assert state["kelvin"] <= 2000
        assert state["brightness_pct"] == 5
        assert state["mode_name"] == "Deep Night"

    def test_4am_interpolates(self):
        state = get_circadian_state(_at(4, 0))
        # Still in the midnight-to-dawn range
        assert state["kelvin"] >= 1800
        assert state["kelvin"] <= 2000


class TestEdgeCases:
    def test_exactly_on_waypoint(self):
        """Exact waypoint time should return exact values."""
        state = get_circadian_state(_at(12, 0))
        assert state["kelvin"] == 6500
        assert state["brightness_pct"] == 100

    def test_returns_time_string(self):
        state = get_circadian_state(_at(14, 30))
        assert state["time"] == "14:30"

    def test_returns_active_lights(self):
        state = get_circadian_state(_at(12, 0))
        assert isinstance(state["active_lights"], list)
        assert len(state["active_lights"]) > 0
