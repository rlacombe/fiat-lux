"""Tests for the lighting scheduler."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from fiat_lux.scheduler import (
    _execute_transition,
    _cleanup_past_jobs,
    _load_schedule,
    _resolve_lights,
    cancel_scheduled,
    list_scheduled,
    schedule_transition,
)


@pytest.fixture
def schedule_file(tmp_path, monkeypatch):
    """Redirect schedule file to tmp dir."""
    f = tmp_path / "schedule.json"
    monkeypatch.setattr("fiat_lux.scheduler.SCHEDULE_FILE", f)
    monkeypatch.setattr("fiat_lux.scheduler.CONFIG_DIR", tmp_path)
    return f


@pytest.fixture
def mock_bridge():
    """Mock the Hue bridge."""
    light1 = MagicMock()
    light1.light_id = 1
    light1.name = "Ceiling lamp 1"

    light2 = MagicMock()
    light2.light_id = 2
    light2.name = "Desk lamp"

    bridge = MagicMock()
    bridge.lights = [light1, light2]

    with patch("fiat_lux.scheduler._get_bridge", return_value=bridge):
        yield bridge


class TestScheduleTransition:
    def test_creates_job(self, schedule_file):
        future = datetime.now() + timedelta(hours=8)
        job_id = schedule_transition(
            start_time=future,
            lights=["all"],
            start_state={"brightness_pct": 1, "kelvin": 2000},
            end_state={"brightness_pct": 100, "kelvin": 5500},
            duration_minutes=20,
            description="Sunrise",
        )
        assert len(job_id) == 8
        jobs = _load_schedule()
        assert len(jobs) == 1
        assert jobs[0]["id"] == job_id
        assert jobs[0]["description"] == "Sunrise"

    def test_multiple_jobs(self, schedule_file):
        future1 = datetime.now() + timedelta(hours=8)
        future2 = datetime.now() + timedelta(hours=22)
        schedule_transition(future1, ["all"], {}, {}, 20, "Morning")
        schedule_transition(future2, ["all"], {}, {}, 30, "Evening")
        jobs = _load_schedule()
        assert len(jobs) == 2


class TestListScheduled:
    def test_filters_past_jobs(self, schedule_file):
        past = datetime.now() - timedelta(hours=2)
        future = datetime.now() + timedelta(hours=2)
        schedule_file.write_text(json.dumps([
            {
                "id": "past1",
                "start_time": past.isoformat(),
                "duration_minutes": 20,
                "lights": ["all"],
            },
            {
                "id": "future1",
                "start_time": future.isoformat(),
                "duration_minutes": 20,
                "lights": ["all"],
            },
        ]))
        pending = list_scheduled()
        assert len(pending) == 1
        assert pending[0]["id"] == "future1"

    def test_empty_schedule(self, schedule_file):
        assert list_scheduled() == []


class TestCancelScheduled:
    def test_cancels_existing(self, schedule_file):
        future = datetime.now() + timedelta(hours=8)
        job_id = schedule_transition(future, ["all"], {}, {}, 20)
        assert cancel_scheduled(job_id)
        assert list_scheduled() == []

    def test_returns_false_for_unknown(self, schedule_file):
        assert cancel_scheduled("nonexistent") is False


class TestResolveLights:
    def test_all_lights(self, mock_bridge):
        ids = _resolve_lights(mock_bridge, ["all"])
        assert len(ids) == 2

    def test_by_name(self, mock_bridge):
        ids = _resolve_lights(mock_bridge, ["Desk lamp"])
        assert ids == [2]

    def test_unknown_name_skipped(self, mock_bridge):
        ids = _resolve_lights(mock_bridge, ["Nonexistent"])
        assert ids == []


class TestExecuteTransition:
    def test_sets_start_then_end_state(self, mock_bridge):
        job = {
            "id": "test1",
            "start_time": datetime.now().isoformat(),
            "lights": ["all"],
            "start_state": {"brightness_pct": 1, "kelvin": 2000},
            "end_state": {"brightness_pct": 100, "kelvin": 5500},
            "duration_minutes": 20,
            "description": "Sunrise",
        }

        with patch("fiat_lux.scheduler.time.sleep"):
            _execute_transition(job)

        calls = mock_bridge.set_light.call_args_list
        # 2 lights x 2 phases (start + end) = 4 calls
        assert len(calls) == 4

        # First two: start state (instant, transitiontime=0)
        start_cmd = calls[0][0][1]
        assert start_cmd["transitiontime"] == 0
        assert start_cmd["on"] is True

        # Last two: end state (with ramp)
        end_cmd = calls[2][0][1]
        assert end_cmd["on"] is True
        # Transition time should be close to 20 min (might be slightly less due to timing)
        assert end_cmd["transitiontime"] > 11000

    def test_caps_transition_at_bridge_max(self, mock_bridge):
        job = {
            "id": "test2",
            "start_time": datetime.now().isoformat(),
            "lights": ["Desk lamp"],
            "start_state": {"brightness_pct": 1},
            "end_state": {"brightness_pct": 100},
            "duration_minutes": 120,  # over the ~109 min limit
            "description": "Long ramp",
        }

        with patch("fiat_lux.scheduler.time.sleep"):
            _execute_transition(job)

        calls = mock_bridge.set_light.call_args_list
        end_cmd = calls[1][0][1]
        assert end_cmd["transitiontime"] == 65535

    def test_catches_up_when_late(self, mock_bridge):
        """If we're 5 minutes late on a 20-minute transition, jump to 25% and ramp the rest."""
        start = datetime.now() - timedelta(minutes=5)
        job = {
            "id": "test3",
            "start_time": start.isoformat(),
            "lights": ["Desk lamp"],
            "start_state": {"brightness_pct": 0, "kelvin": 2000},
            "end_state": {"brightness_pct": 100, "kelvin": 5500},
            "duration_minutes": 20,
            "description": "Late sunrise",
        }

        with patch("fiat_lux.scheduler.time.sleep"):
            _execute_transition(job)

        calls = mock_bridge.set_light.call_args_list
        # First call: jump to interpolated position (~25%)
        now_cmd = calls[0][0][1]
        assert now_cmd["transitiontime"] == 0
        assert now_cmd["bri"] > 1  # should be partway through
        # Second call: ramp remaining ~15 minutes
        end_cmd = calls[1][0][1]
        assert end_cmd["transitiontime"] < 20 * 60 * 10  # less than full 20 min


class TestCleanupPastJobs:
    def test_removes_completed_jobs(self, schedule_file):
        past = datetime.now() - timedelta(hours=2)
        future = datetime.now() + timedelta(hours=2)
        schedule_file.write_text(json.dumps([
            {"id": "old", "start_time": past.isoformat(), "duration_minutes": 20},
            {"id": "new", "start_time": future.isoformat(), "duration_minutes": 20},
        ]))
        _cleanup_past_jobs()
        jobs = _load_schedule()
        assert len(jobs) == 1
        assert jobs[0]["id"] == "new"
