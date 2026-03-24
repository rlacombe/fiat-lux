"""Tests for the GUI menubar app module."""

from unittest.mock import patch, MagicMock

import pytest

rumps = pytest.importorskip("rumps")

from heylux.gui import HeyLuxApp, ICON_IDLE, ICON_LISTENING, ICON_PROCESSING


class TestHeyLuxApp:
    def test_creates_app(self):
        app = HeyLuxApp()
        assert app.name == "Hey Lux"
        assert app.title == ICON_IDLE

    def test_has_menu_items(self):
        app = HeyLuxApp()
        assert len(app.menu) >= 3

    def test_set_status(self):
        app = HeyLuxApp()
        app._set_status(ICON_LISTENING)
        assert app.title == ICON_LISTENING
        app._set_status(ICON_PROCESSING)
        assert app.title == ICON_PROCESSING


class TestDaemonManagement:
    @patch("heylux.gui.PID_FILE")
    def test_daemon_not_running_no_pid(self, mock_pid):
        from heylux.gui import _daemon_running
        mock_pid.exists.return_value = False
        assert _daemon_running() is False


class TestSendToDaemon:
    def test_send_is_callable(self):
        """Verify the function exists and is callable."""
        from heylux.gui import _send_to_daemon
        assert callable(_send_to_daemon)
